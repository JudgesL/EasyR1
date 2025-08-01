from distutils.command.config import config
from importlib.metadata import SelectableGroups
from transformers import Qwen3PreTrainedModel, Qwen3Model, Qwen3ForCausalLM, PreTrainedModel, BertModel, BertPreTrainedModel, BertForMaskedLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class CausalLMOutputWithPastAndCTR(CausalLMOutputWithPast):
    ctr_pred: Optional[torch.Tensor] = None
    cvr_pred: Optional[torch.Tensor] = None
    is_cvr_sample: Optional[torch.Tensor] = None


class CTRCVRHead(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            raise ValueError("hidden_size is not defined in config")
            
        self.dnn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1),  # Output layer for prediction
            nn.Sigmoid()
        )
        
        for layer in self.dnn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, hidden_states):
        return self.dnn(hidden_states)

class QwenWithCTRCVR(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.ctr_head = CTRCVRHead(config)
        self.cvr_head = CTRCVRHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None, 
        **kwargs
    ):
        # model output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state

        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            next_sent_feat = hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            next_sent_feat = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        ctr_pred = self.ctr_head(next_sent_feat)
        cvr_pred = self.cvr_head(next_sent_feat)

        device = ctr_pred.device
        loss = torch.tensor(0.0, device=device)
        ctr_loss = torch.tensor(0.0, device=device)
        cvr_loss = torch.tensor(0.0, device=device)

        # get ctr_label, cvr_label, is_cvr_sample, is_use_cvr_loss from kwargs
        ctr_label = kwargs.get("ctr_label", None)
        cvr_label = kwargs.get("cvr_label", None)
        is_cvr_sample = kwargs.get("is_cvr_sample", None)
        is_use_cvr_loss = kwargs.get("is_use_cvr_loss", None)

        print_detail_logs_for_debug = 0
        if print_detail_logs_for_debug:
            # 打印ctr_pred每一条
            print("[LOG] ctr_pred (batch前10条):")
            print(np.round(ctr_pred.detach().cpu().float().numpy()[:10], 6).reshape(-1))

            # 打印ctr_label
            if ctr_label is not None:
                print("[LOG] ctr_label (batch前10条):")
                print(ctr_label.detach().cpu().float().numpy()[:10].reshape(-1))

            # 打印next_sent_feat的前5个样本的前10维（较大batch建议缩小到前3个样本）
            next_sent_feat_np = next_sent_feat.data.detach().cpu().float().numpy()
            print("[LOG] next_sent_feat (前5个样本前10维):")
            for i in range(min(5, next_sent_feat_np.shape[0])):
                print(f"  sample {i}: {np.round(next_sent_feat_np[i, :10], 4)}")

            print("[LOG] input_ids (前5个样本):")
            for i in range(min(5, input_ids.shape[0])):
                print(f"  sample {i}: {input_ids[i].cpu().numpy().tolist()}")

        # deal CTR loss
        if ctr_label is not None:
            ctr_label = ctr_label.to(ctr_pred.dtype)
            ctr_pred = ctr_pred.view(-1)
            ctr_label = ctr_label.view(-1)
            ctr_loss = F.binary_cross_entropy(ctr_pred, ctr_label, reduction='none')

        # calculate CVR loss
        if (cvr_label is not None) and (is_cvr_sample is not None):
            cvr_mask = is_cvr_sample.bool()
            batch_size = ctr_loss.size(0)  # get batch size

            # initialize CVR loss to 0
            cvr_loss_full = torch.zeros(batch_size, device=device, dtype=cvr_pred.dtype)
            
            if cvr_mask.any():
                # get masked sample CVR prediction and label
                cvr_pred_masked = cvr_pred[cvr_mask].view(-1)
                cvr_label_masked = cvr_label[cvr_mask].to(cvr_pred.dtype).view(-1)
                
                # calculate CVR loss
                cvr_loss = F.binary_cross_entropy(
                    cvr_pred_masked,
                    cvr_label_masked,
                    reduction='none'
                )
                
                # fill effective loss to full loss tensor
                cvr_loss_full[cvr_mask] = cvr_loss
            else:
                # if no masked sample, cvr_loss_full remains 0
                pass
        else:
            # if not provide CVR label or is_cvr_sample, cvr_loss_full remains 0
            cvr_loss_full = torch.zeros_like(ctr_loss, device=device)

        # calculate total loss
        total_loss = ctr_loss + cvr_loss_full  
        loss_mean = total_loss.mean()

        return CausalLMOutputWithPastAndCTR(
            loss=loss_mean,
            logits=torch.stack([ctr_pred.squeeze(-1), cvr_pred.squeeze(-1)], dim=1) ,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            ctr_pred=ctr_pred,
            cvr_pred=cvr_pred,
            is_cvr_sample=is_cvr_sample
        )

class CTRCVRHeadNoSigmoid(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            raise ValueError("hidden_size is not defined in config")
            
        self.dnn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)  # Output layer for prediction
        )
        
        for layer in self.dnn:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, hidden_states):
        return self.dnn(hidden_states)

class BertWithCTRCVR(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.vocab_size = self.config.vocab_size
        self.ctr_head = CTRCVRHead(self.config)
        self.cvr_head = CTRCVRHead(self.config)
        self.post_init()

    @classmethod
    def from_config(cls, config, trust_remote_code=False):
        return cls(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None, 
        ctr_label= None,
        cvr_label= None,
        is_cvr_sample= None,
        is_use_cvr_loss= None,
        **kwargs
    ):
        # model output
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = outputs.last_hidden_state

        next_sent_feat = hidden_states[:, 0]  # 取[CLS]的hidden state
        
        # left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        # if left_padding:
        #     next_sent_feat = hidden_states[:, -1]
        # else:
        #     sequence_lengths = attention_mask.sum(dim=1) - 1
        #     batch_size = hidden_states.shape[0]
        #     next_sent_feat = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

        ctr_pred = self.ctr_head(next_sent_feat)
        cvr_pred = self.cvr_head(next_sent_feat)

        device = ctr_pred.device
        loss = torch.tensor(0.0, device=device)
        ctr_loss = torch.tensor(0.0, device=device)
        cvr_loss = torch.tensor(0.0, device=device)

        print_detail_logs_for_debug = 0
        if print_detail_logs_for_debug:
            print("**************Debug Info******************")
            print("[LOG] hidden_states:", np.round(hidden_states[0, :10].detach().cpu().float().numpy(), 6))

            print("[LOG] input_ids (前5个样本):")
            for i in range(min(5, input_ids.shape[0])):
                print(f"  sample {i}: {input_ids[i].cpu().numpy().tolist()}")

            # 打印next_sent_feat的前5个样本的前10维（较大batch建议缩小到前3个样本）
            next_sent_feat_np = next_sent_feat.data.detach().cpu().float().numpy()
            print("[LOG] next_sent_feat (前5个样本前10维):")
            for i in range(min(5, next_sent_feat_np.shape[0])):
                print(f"  sample {i}: {np.round(next_sent_feat_np[i, :10], 4)}")

            # 打印ctr_label
            if ctr_label is not None:
                print("[LOG] ctr_label (batch前10条):")
                print(ctr_label.detach().cpu().float().numpy()[:10].reshape(-1))
            
            # 打印ctr_pred每一条
            print("[LOG] ctr_pred (batch前10条):")
            print(np.round(ctr_pred.detach().cpu().float().numpy()[:10], 6).reshape(-1))

            # 打印经过 sigmoid 后的结果
            ctr_sigmoid = torch.sigmoid(ctr_pred)
            print("[LOG] ctr_pred sigmoid后 (batch前10条):")
            print(np.round(ctr_sigmoid.detach().cpu().float().numpy()[:10], 6).reshape(-1))

        # deal CTR loss
        if ctr_label is not None:
            ctr_label = ctr_label.to(ctr_pred.dtype)
            ctr_pred = ctr_pred.view(-1)
            ctr_label = ctr_label.view(-1)
            ctr_loss = F.binary_cross_entropy(ctr_pred, ctr_label, reduction='none')
            ctr_loss = ctr_loss.to(ctr_pred.dtype)

        # calculate CVR loss
        if (cvr_label is not None) and (is_cvr_sample is not None):
            cvr_mask = is_cvr_sample.bool()
            batch_size = ctr_loss.size(0)  # get batch size

            # initialize CVR loss to 0
            cvr_loss_full = torch.zeros(batch_size, device=device, dtype=cvr_pred.dtype)
            
            if cvr_mask.any():
                # get masked sample CVR prediction and label
                cvr_pred_masked = cvr_pred[cvr_mask].view(-1)
                cvr_label_masked = cvr_label[cvr_mask].to(cvr_pred.dtype).view(-1)
                
                # calculate CVR loss
                cvr_loss = F.binary_cross_entropy(
                    cvr_pred_masked,
                    cvr_label_masked,
                    reduction='none'
                )
                cvr_loss = cvr_loss.to(cvr_loss_full.dtype)
                # fill effective loss to full loss tensor
                
                cvr_loss_full[cvr_mask] = cvr_loss
            else:
                # if no masked sample, cvr_loss_full remains 0
                cvr_loss_full = 0.0 * cvr_pred.sum() + torch.zeros_like(ctr_loss, device=device)
        else:
            # if not provide CVR label or is_cvr_sample, cvr_loss_full remains 0
            cvr_loss_full = 0.0 * cvr_pred.sum() + torch.zeros_like(ctr_loss, device=device)

        # calculate total loss
        total_loss = ctr_loss + cvr_loss_full  
        loss_mean = total_loss.mean()
        # loss_mean = loss_mean + 0.0 * sum(p.sum() for p in self.parameters())
        logits =torch.stack([ctr_pred.squeeze(-1), cvr_pred.squeeze(-1)], dim=1)

        return {
            "loss": loss_mean,
            "logits": logits,
            "ctr_pred": ctr_pred,
            "cvr_pred": cvr_pred,
            "is_cvr_sample": is_cvr_sample,
        }