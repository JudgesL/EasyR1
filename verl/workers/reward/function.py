# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict
from .model import QwenWithCTRCVR,BertWithCTRCVR
import torch
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
from ...protocol import DataProto
from .config import RewardConfig
import jieba
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RewardInput(TypedDict):
    response: str
    response_length: int
    ground_truth: str
    refer_message: str


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


SequentialRewardFunction = Callable[[RewardInput], RewardScore]

BatchRewardFunction = Callable[[List[RewardInput]], List[RewardScore]]

HybridLLMRuleRewardFunction = Callable[[RewardInput], RewardScore]


class FunctionRewardManager(ABC):
    """Reward manager for rule-based reward."""

    def __init__(self, config: RewardConfig, tokenizer: PreTrainedTokenizer):
        if config.reward_function is None:
            raise ValueError("Reward function is not provided.")

        if not os.path.exists(config.reward_function):
            raise FileNotFoundError(f"Reward function file {config.reward_function} not found.")

        spec = importlib.util.spec_from_file_location("custom_reward_fn", config.reward_function)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_reward_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load reward function: {e}")

        if not hasattr(module, config.reward_function_name):
            raise AttributeError(f"Module {module} does not have function {config.reward_function_name}.")

        reward_fn = getattr(module, config.reward_function_name)
        print(f"Using reward function `{config.reward_function_name}` from `{config.reward_function}`.")
        self.reward_fn = partial(reward_fn, **config.reward_function_kwargs)
        self.config = config
        self.tokenizer = tokenizer

    @abstractmethod
    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        """Compute reward for a batch of data."""
        ...

class SequentialFunctionRewardManager(FunctionRewardManager):
    reward_fn: SequentialRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            score = self.reward_fn(
                {
                    "response": response_str,
                    "response_length": response_length[i],
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics


class BatchFunctionRewardManager(FunctionRewardManager):
    reward_fn: BatchRewardFunction

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_inputs = []
        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        for i in range(len(data)):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            reward_inputs.append(
                {
                    "response": response_str,
                    "response_length": response_length[i],
                    "ground_truth": data.non_tensor_batch["ground_truth"][i],
                }
            )

        scores = self.reward_fn(reward_inputs)
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        for i, score in enumerate(scores):
            reward_tensor[i, response_length[i] - 1] = score["overall"]
            for key, value in score.items():
                reward_metrics[key].append(value)

        return reward_tensor, reward_metrics

class HybridLLMRuleRewardManager(FunctionRewardManager):
    reward_fn: HybridLLMRuleRewardFunction
    def __init__(self, config: RewardConfig, tokenizer):
        super().__init__(config, tokenizer)
        print("[DEBUG] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("[DEBUG] torch.cuda.is_available():", torch.cuda.is_available())
        print("[DEBUG] torch.cuda.device_count():", torch.cuda.device_count())
        self.reward_model_batch_size = getattr(config, "reward_model_batch_size", 1024)
        self.reword_weight = getattr(config, "reward_model_weight", 0.5)
        self.rule_weight = getattr(config, "rule_weight", 0.5)
        self.diversity_weight = getattr(config, "diversity_weight", 0.5)
        self.model_score_type = getattr(config, "model_score_type", "ctcvr")
        self.model_q_process = getattr(config, "model_q_process", "log")
        model_path = getattr(config, "reward_model_path", None)
        if model_path:
            print(f"[HybridLLMRuleRewardManager] Loading reward LLM from {model_path}")
            special_tokens_dict = {'additional_special_tokens': ['[SEP]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_toks} special tokens")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm_model = QwenWithCTRCVR.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
            self.llm_model = self.llm_model.to(torch.float16)
            self.llm_model.eval()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.llm_model.to(device)
            if device == "cuda":
                print("[DEBUG] Model device:", next(self.llm_model.parameters()).device)
                print("[DEBUG] Model CUDA device id:", next(self.llm_model.parameters()).device.index)
            else:
                print("[DEBUG] Model device: cpu")
        else:
            print("[HybridLLMRuleRewardManager] No LLM reward_model_path provided. Only rule-based reward will be used.")
            self.llm_tokenizer = None
            self.llm_model = None

    def jieba_tokenize(self, text):
        return jieba.lcut(text)
    
    def extract_model_output(self, model_output: str, refer_message: str) -> str:
        """
        从模型输出的文本中提取出需要的信息，包括：
        【卡片标题】：{title}
        【要点】：{question}
        【回答】：{answer}
        参考信息格式：
        pagetitle__bidword
        """
        # 解析卡片信息
        title, question, answer = '', '', ''
        for line in model_output.split('\n'):
            if line.startswith('【卡片标题】：'):
                title = line.replace('【卡片标题】：', '').strip()
            elif line.startswith('【要点】：'):
                question = line.replace('【要点】：', '').strip()
            elif line.startswith('【回答】：'):
                answer = line.replace('【回答】：', '').strip()
        # 解析参考信息
        if '__' in refer_message:
            pagetitle, bidword  = refer_message.split('__', 1)
        return title, question, answer, pagetitle, bidword
    
    def model_output_to_reward_input_qwen(self, title, question, answer, pagetitle, bidword) -> str:
        """
        参考信息格式：
        pagetitle__bidword
        转换为 reward 可接受的格式：
        f"你是一位精通用户心理与商业广告的专家，请你根据用户query和广告内容，预估用户是否会点击广告并发生转化。"
                f"query:{pagetitle}，"
                f"广告内容：广告拍卖词:{bidword}，广告标题「{title}」广告问答卡创意问「{ask_content}」回答「{answer_content}」"
        """
        # 拼接reward输入格式
        reward_input = (
            "你是一位精通用户心理与商业广告的专家，请你根据用户query和广告内容，预估用户是否会点击广告并发生转化。"
            f"query:{pagetitle}，"
            f"广告内容：广告拍卖词:{bidword}，广告标题「{title}」广告问答卡创意问「{question}」回答「{answer}」"
        )
        return reward_input

    def calc_overall_reward(self, rule_score, model_score, diversity_score):
        return self.reword_weight * model_score + self.rule_weight * rule_score + self.diversity_weight * diversity_score

    def get_model_score(self, response_str):
        inputs = self.llm_tokenizer(
            response_str, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.llm_model(**inputs)
            ctr_score = out.logits[:, 0].item()
        return ctr_score

    def compute_reward(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        import time
        from collections import defaultdict
        print_debug_log = 0

        def log(msg, level="DEBUG"):
            if print_debug_log:
                print(f"[{level}] compute_reward: {msg}")

        start_time = time.time()
        log("start compute_reward (batch version)")

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)

        response_ids = data.batch["responses"]
        response_length = data.batch["response_mask"].sum(dim=-1)
        batch_size = len(data)

        # Step 1: 先计算 rule-based reward
        rule_scores = []
        reward_model_inputs = []
        all_rule_score_dicts = []
        titles = []
        questions = []
        answers = []
        for i in range(batch_size):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )

            gt = data.non_tensor_batch["ground_truth"][i]
            refer_msg = data.non_tensor_batch["refer_message"][i]
            rule_score_dict = self.reward_fn({
                "response": response_str,
                "response_length": int(response_length[i]),
                "ground_truth": gt,
                "refer_message": refer_msg,
            })
            rule_score = rule_score_dict["overall"]
            rule_scores.append(rule_score)
            all_rule_score_dicts.append(rule_score_dict)  # 保存全部

            # 为模型推理、多样性奖励准备输入
            if self.llm_model is not None:
                title, question, answer, pagetitle, bidword = self.extract_model_output(response_str, refer_msg)
                reward_input = self.model_output_to_reward_input_qwen(title, question, answer, pagetitle, bidword)
                reward_model_inputs.append(reward_input)
                titles.append(title)
                questions.append(question)
                answers.append(answer)
            else:
                reward_model_inputs.append(None)
                titles.append(None)
                questions.append(None)
                answers.append(None)

            log(f"sample {i}: rule_score={rule_score}, response_len={int(response_length[i])}")

        # Step 2: batch 模型推理
        model_scores = [0.0] * batch_size
        if self.llm_model is not None:
            valid_indices_and_inputs = [(i, x) for i, x in enumerate(reward_model_inputs) if x is not None]
            if valid_indices_and_inputs:
                batch_size_ = self.reward_model_batch_size
                for start in range(0, len(valid_indices_and_inputs), batch_size_):
                    end = start + batch_size_
                    batch = valid_indices_and_inputs[start:end]
                    batch_indices = [idx for idx, _ in batch]
                    batch_inputs = [x for _, x in batch]
                    log(f"model forward batch: {start}-{end}, size={len(batch_inputs)}")
                    enc = self.llm_tokenizer(
                        batch_inputs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    enc = {k: v.to(self.llm_model.device) for k, v in enc.items()}
                    with torch.no_grad():
                        out = self.llm_model(**enc)
                        batch_ctr_scores = out.logits[:, 0].cpu().tolist()
                        batch_ctcvr_scores = out.logits[:, 1].cpu().tolist()
                    if self.model_score_type == "ctcvr":
                        # ctcvr数值过小，处理时先*100
                        for idx, score in zip(batch_indices, batch_ctcvr_scores):
                            if self.model_q_process == "log":
                                model_scores[idx] = math.log(max(score * 100, 0.0) + 1.0)
                            else:
                                model_scores[idx] = score * 100
                            log(f"sample {idx}: model_score={score}")
                    else:
                        for idx, score in zip(batch_indices, batch_ctr_scores):
                            if self.model_q_process == "log":
                                model_scores[idx] = math.log(max(score, 0.0) + 1.0)
                            else:
                                model_scores[idx] = score
                            log(f"sample {idx}: model_score={score}")
            else:
                log("no valid model inputs, skip model scoring")

        # Step 3: 多样性奖励
        tfidf_vectorizer = TfidfVectorizer(tokenizer=self.jieba_tokenize)
        diversity_score_dicts = []  # 每个样本的多样性得分字典
        if batch_size > 1:
            title_tfidf_matrix = tfidf_vectorizer.fit_transform(titles)
            question_tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
            answer_tfidf_matrix = tfidf_vectorizer.fit_transform(answers)
            # 标题之间两两计算similarity
            title_similarity_matrix = cosine_similarity(title_tfidf_matrix)
            question_similarity_matrix = cosine_similarity(question_tfidf_matrix)
            answer_similarity_matrix = cosine_similarity(answer_tfidf_matrix)
            for i in range(batch_size):
                # similarity_matrix[i] 是第 i 条 response 对所有 response 的相似度；
                # similarity_matrix[i].sum() 是与所有 response 的相似度和；
                # - 1.0 是去掉与自己相似度 1.0；/ (batch_size - 1) 是求平均相似度；1.0 - avg_sim 代表越不相似越高分（多样性越好）。
                title_similarity = (title_similarity_matrix[i].sum() - 1.0) / (batch_size - 1)
                question_similarity = (question_similarity_matrix[i].sum() - 1.0) / (batch_size - 1)
                answer_similarity = (answer_similarity_matrix[i].sum() - 1.0) / (batch_size - 1)
                # 多样性 = 1 - 平均相似度
                diversity_dict = {
                    "title_diversity_score": 1.0 - title_similarity,
                    "question_diversity_score": 1.0 - question_similarity,
                    "answer_diversity_score": 1.0 - answer_similarity,
                }
                diversity_score_dicts.append(diversity_dict)
        else:
            # batch_size == 1 时，定义多样性得分为0.0
            for _ in range(batch_size):
                diversity_dict = {
                    "title_diversity_score": 0.0,
                    "question_diversity_score": 0.0,
                    "answer_diversity_score": 0.0,
                }
                diversity_score_dicts.append(diversity_dict)

        # Step 4: 计算 overall reward
        for i in range(batch_size):
            diversity_score = (diversity_score_dicts[i]["answer_diversity_score"] + diversity_score_dicts[i]["question_diversity_score"] + diversity_score_dicts[i]["title_diversity_score"]) / 3
            overall = self.calc_overall_reward(rule_scores[i], model_scores[i], diversity_score)
            reward_tensor[i, response_length[i] - 1] = overall
            reward_metrics["overall"].append(overall)
            reward_metrics["rule_score"].append(rule_scores[i])
            reward_metrics["model_score"].append(model_scores[i])
            # 加入 rule_score_dict 内容
            rule_score_dict = all_rule_score_dicts[i]
            for key, value in rule_score_dict.items():
                if key not in ("overall",):  # overall已加
                    reward_metrics[key].append(value)

            # 加入 diversity_score_dict 内容
            diversity_score_dict = diversity_score_dicts[i]
            for key, value in diversity_score_dict.items():
                reward_metrics[key].append(value)

            log(f"sample {i}: overall={overall}")

        log(f"finished compute_reward, total_time={time.time() - start_time:.3f}s")
        return reward_tensor, reward_metrics