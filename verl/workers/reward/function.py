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
from transformers import AutoTokenizer, AutoModelForCausalLM
from ...protocol import DataProto
from .config import RewardConfig
import jieba
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import Counter
from .prompts import PROMPT_BASE
from vllm import LLM, SamplingParams

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
        # ctcvr model 模型相关
        model_path = getattr(config, "reward_model_path", None)
        self.reward_model_batch_size = getattr(config, "reward_model_batch_size", 1024)
        self.model_score_type = getattr(config, "model_score_type", "ctcvr")
        self.model_q_process = getattr(config, "model_q_process", "log")

        # judge model 相关
        judge_model_path = getattr(config, "judge_model_path", None)
        self.judge_model_batch_size = getattr(config, "judge_model_batch_size", 256)
        self.relevant_weight = getattr(config, "relevant_weight", 0.5)
    
        # reward 权重相关
        self.reword_weight = getattr(config, "reward_model_weight", 0.5)
        self.rule_weight = getattr(config, "rule_weight", 0.5)
        self.diversity_weight = getattr(config, "diversity_weight", 0.5)

        # 多样性reward相关
        self.n_gram_low_bound= getattr(config, "n_gram_low_bound", 2)
        self.n_gram_up_bound = getattr(config, "n_gram_up_bound", 5)
        self.n_gram_threshold = getattr(config, "n_gram_threshold", 0.5)
        self.ngram_penalty = getattr(config, "ngram_penalty", 0.1)

        # 黑名单相关
        black_list_path = getattr(config, "black_list_path", None)
        # 先启动VLLM，并保证其先用掉1张卡，不然VLLM会自动用掉所有的卡
        if judge_model_path:
            print(f"[HybridLLMRuleRewardManager] Loading judge model from {judge_model_path}")
            # self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
            # self.judge_model = AutoModelForCausalLM.from_pretrained(
            #     judge_model_path,
            #     torch_dtype=torch.float16,
            #     device_map="auto"
            # )
            # self.judge_model.eval()
            self.judge_model = LLM(
                model=judge_model_path,
                dtype="float16",              # 推荐用float16提升速度
                tensor_parallel_size=1,
                # trust_remote_code=True      # 有的模型需要
            )
            self.vllm_sampling_params = SamplingParams(
                max_tokens=100,              # 你要生成的最大token数
                temperature=0.5,
                top_p=1.0,
            )
        else:
            print("[HybridLLMRuleRewardManager] No judge model path provided.")
            self.judge_tokenizer = None
            self.judge_model = None

        if model_path:
            print(f"[HybridLLMRuleRewardManager] Loading reward LLM from {model_path}")
            special_tokens_dict = {'additional_special_tokens': ['[SEP]']}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_toks} special tokens")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_path)
            # 明确指定reward model到cuda:1
            self.llm_model = QwenWithCTRCVR.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")
            self.llm_model.eval()
            reward_model_device = torch.device("cuda:1") if torch.cuda.is_available() and torch.cuda.device_count() > 1 else torch.device("cuda:0")
            self.llm_model = self.llm_model.to(reward_model_device)
            if reward_model_device:
                print("[DEBUG] Model device:", next(self.llm_model.parameters()).device)
                print("[DEBUG] Model CUDA device id:", next(self.llm_model.parameters()).device.index)
            else:
                print("[DEBUG] Model device: cpu")
        else:
            print("[HybridLLMRuleRewardManager] No LLM reward_model_path provided. Only rule-based reward will be used.")
            self.llm_tokenizer = None
            self.llm_model = None

        if black_list_path:
            print(f"[HybridLLMRuleRewardManager] Loading black list from {black_list_path}")
            with open(black_list_path, 'r', encoding='gbk') as f:
                self.black_list = set([line.strip() for line in f])
        else:
            print("[HybridLLMRuleRewardManager] No black list path provided.")
            self.black_list = None


    def map_char_rewards_to_tokens(self,
        response_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        response_str: str,
        char_rewards: List[float],
    ) -> torch.Tensor:
        """
        将“逐字符（char-level）分数”映射为“逐 token 分数”。                                                          

        约定与保证：
        - char_rewards 的长度必须等于 response_str 的字符长度（Python 字符串的 len 计数）。
        - 返回的向量长度与 response_ids 完全相同；对特殊 token（BOS/EOS/PAD 等）填 0。
        - 对于覆盖多字符的 token，将其覆盖区间内的字符分数取平均值。

        参数:
            response_ids: torch.Tensor
                单条响应的 token id 序列（建议传入有效部分，形如 [T1, T2, ..., Tn]）。
            tokenizer: PreTrainedTokenizer
                与生成 response_ids 相同的 tokenizer。
            response_str: str
                该响应解码后的完整明文（和你打分的对象一致）。
            char_rewards: List[float]
                与 response_str 等长的分数序列（每个字符一个分数）。

        返回:
            torch.Tensor
                与 response_ids 等长的 token 级分数张量，dtype=float32，device 与 response_ids 相同。
        """
        # ---------- 基本校验 ----------
        # 保证字符级分数长度与明文长度完全一致，否则无法一一对应。
        if len(response_str) != len(char_rewards):
            raise ValueError(
                f"[map_char_rewards_to_tokens] Length mismatch: "
                f"len(response_str)={len(response_str)} vs len(char_rewards)={len(char_rewards)}"
            )

        device = response_ids.device
        ids_list = response_ids.tolist()

        # 输出张量，初始化为 0（对 special token 会保留为 0；普通 token 之后填平均分）
        token_rewards = torch.zeros_like(response_ids, dtype=torch.float32)

        # ---------- 标记特殊 token ----------
        # special_mask: 与 ids_list 等长，特殊 token 位置为 1；普通 token 为 0
        # already_has_special_tokens=True 表示 ids_list 可能包含 special，需要直接判断。
        try:
            special_mask = tokenizer.get_special_tokens_mask(
                ids_list, already_has_special_tokens=True
            )
        except Exception:
            # 某些自定义 tokenizer 可能不支持该函数；保守起见默认全为非特殊
            special_mask = [0] * len(ids_list)

        # 记录普通 token 的下标（需要写入分数的位置）
        non_special_indices = [i for i, m in enumerate(special_mask) if m == 0]

        # ---------- Fast 路径：使用 offset mapping ----------
        # 条件：tokenizer 为 Fast 版本（底层为 tokenizers 库），支持返回每个 token 对应的字符区间
        used_fast_path = False
        # if getattr(tokenizer, "is_fast", False):
        #     try:
        #         # 对 response_str 重新分词，且不引入任何 special tokens
        #         # 这样得到的 input_ids 和 offset_mapping 是原始明文的纯文本切分
        #         enc = tokenizer(
        #             response_str,
        #             add_special_tokens=False,
        #             return_offsets_mapping=True
        #         )
        #         text_token_ids = enc["input_ids"]                # 纯文本 token id 序列
        #         offsets = enc["offset_mapping"]                  # 每个 token 对应的 (start, end) 字符下标（左闭右开）

        #         # 将每个文本 token 的字符区间分数取平均
        #         text_token_rewards: List[float] = []
        #         for (start, end) in offsets:
        #             if end > start:
        #                 # 平均该区间内的字符分数；注意这里的切片是基于 Python 字符的下标
        #                 span = char_rewards[start:end]
        #                 text_token_rewards.append(float(sum(span)) / (end - start))
        #             else:
        #                 # 部分 tokenizer 可能产生零宽 token（极少见）；安全起见置 0
        #                 text_token_rewards.append(0.0)

        #         # 将 response_ids 去掉 special 后得到的 id 序列，与 text_token_ids 对齐：
        #         # 如果完全一致，说明两边切分一致，可直接把 text_token_rewards 写回到非特殊 token 位置。
        #         non_special_ids = [ids_list[i] for i in non_special_indices]
        #         if non_special_ids == text_token_ids:
        #             for out_pos, r in zip(non_special_indices, text_token_rewards):
        #                 token_rewards[out_pos] = r
        #             used_fast_path = True
        #         # 若不一致，说明重分词与原始 ids 有轻微差异（可能来自空白处理等），转入慢路径兜底。
        #     except Exception:
        #         # 某些 tokenizer/输入组合可能不支持 offsets；直接转慢路径
        #         used_fast_path = False

        # ---------- Slow 路径（兜底）：解码游标法 ----------
        # 思路：逐 token 解码成字符串片段 piece，然后用一个 cursor 在 response_str 上滑动，
        # 取与 piece 长度相等的字符分数做平均。这样无需 offsets，也能对齐大多数 BPE/SentencePiece 行为。
        if not used_fast_path:
            cursor = 0  # 指向 response_str 中尚未消费的起始位置
            n_chars = len(response_str)

            for idx, tid in enumerate(ids_list):
                if special_mask[idx] == 1:
                    # 特殊 token：保持 0 分并跳过
                    token_rewards[idx] = 0.0
                    continue

                # 解码当前单个 token（skip_special_tokens=True 避免把特殊符号文本解出来）
                piece = tokenizer.decode([tid], skip_special_tokens=True)

                # 片段为空串的情况（偶尔会出现）：给 0 分，不移动光标
                if not piece:
                    token_rewards[idx] = 0.0
                    continue

                piece_len = len(piece)

                # 理想情况下：response_str[cursor : cursor + piece_len] 恰好等于 piece。
                # 为了保持 O(N) 线性时间，这里不做复杂的查找/对齐，仅按长度切分。
                # 这在 GPT2 系列（空白以空格还原）、SentencePiece（'▁' 还原为空格）等常见 tokenizer 中可用。
                start = cursor
                end = min(cursor + piece_len, n_chars)

                if start >= n_chars:
                    # 安全保护：游标已到达尾部（这通常意味着上游长度不一致）
                    token_rewards[idx] = 0.0
                else:
                    span = char_rewards[start:end]
                    if span:
                        token_rewards[idx] = float(sum(span)) / len(span)
                    else:
                        token_rewards[idx] = 0.0

                # 按 piece 的可见字符长度推进游标
                cursor = end

            # 可选的健壮性检查（不抛错，只在你需要时日志提示）：
            # if cursor != n_chars:
            #     print(f"[map_char_rewards_to_tokens] Warning: cursor({cursor}) != len(response_str)({n_chars}).")

        return token_rewards.to(device)

    def get_ngrams(self, text, n):
        return [text[i:i + n] for i in range(len(text) - n + 1)]

    def remove_specific_punctuation(self, text):
        return re.sub(r'[、（）()，,。.!！;；“”‘’""\'\']', '', text)

    def is_order_ngram(self, gram):
        return any(order in gram for order in ["第一", "第二", "第三", "第四"])
    
    def calc_high_freq_multi_ngram_penalty(self, batch_sentences, stopwords=None):
        ngramed_sentences = []
        for (sent, start_idx) in batch_sentences:
            clean_sent = self.remove_specific_punctuation(sent)
            ngrams = []
            for n in range(self.n_gram_low_bound, self.n_gram_up_bound+1):
                ngrams.extend(self.get_ngrams(clean_sent, n))
            if stopwords:
                ngrams = [gram for gram in ngrams if not any(sw in gram for sw in stopwords)]
            ngramed_sentences.append(ngrams)

        batch_size = len(ngramed_sentences)
        ngram_in_sentence = Counter()
        for ngrams in ngramed_sentences:
            for gram in set(ngrams):
                ngram_in_sentence[gram] += 1

        high_freq_ngrams = {
            gram: cnt / batch_size
            for gram, cnt in ngram_in_sentence.items()
            if cnt / batch_size >= 0.15  # 至少比最低阈值高才纳入打分
        }
        # 去除顺序词组
        high_freq_ngrams = {
            gram: ratio for gram, ratio in high_freq_ngrams.items()
            if not self.is_order_ngram(gram)
        }
        return high_freq_ngrams  # dict: {gram:频率}

    def get_ngram_char_level_reward(self, extracted_sentence, high_freq_ngrams, char_score):
        """
        给句子中出现的高频 n-gram 打分。
        high_freq_ngrams: dict[str, float] -> {ngram: ratio}
        char_score: 全局字符分数向量
        """
        sent, idx = extracted_sentence
        length = len(sent)
        sent_score = [0.0] * length
        if length == 0:
            return char_score, 0.0

        def score_from_ratio(ratio: float) -> float:
            """根据 ratio 映射分数"""
            if ratio > 0.25:
                return -1.0
            elif ratio > 0.20:
                return -0.5
            elif ratio > 0.15:
                return -0.3
            return 0.0

        # 遍历 ngram
        for gram, ratio in high_freq_ngrams.items():
            gram_len = len(gram)
            if gram_len == 0:
                continue

            gram_score = score_from_ratio(ratio)
            if gram_score == 0.0:
                continue  # 没有惩罚就跳过

            # 在句子中查找所有位置
            start = 0
            while True:
                pos = sent.find(gram, start)
                if pos == -1:
                    break
                for k in range(gram_len):
                    if pos + k < length:
                        sent_score[pos + k] = min(sent_score[pos + k], gram_score)  # 保留 min
                start = pos + 1

        # 映射到全局 char_score
        for i in range(length):
            global_idx = i + idx
            # 如果出现问题，打印debug信息
            if global_idx >= len(char_score):
                print("[DEBUG] char_score out of range!")
                print(f"  sent: {sent!r}")
                print(f"  idx: {idx}")
                print(f"  sent_score: {sent_score}")
                print(f"  high_freq_ngrams: {high_freq_ngrams}")
                print(f"  char_score(len={len(char_score)}): {char_score}")
                print(f"  global_idx: {global_idx}, i: {i}")
                return char_score, sum(sent_score) / length + 1
            # 否则赋值
            char_score[global_idx] += sent_score[i]
        return char_score, sum(sent_score) / length + 1

    def get_black_list_char_level_reward(self, extracted_sentence, char_score):
        """
        给句子中出现的黑名单词打分。
        black_list: set/list[str] 黑名单词
        char_score: 全局字符分数向量
        """
        sent, idx = extracted_sentence
        length = len(sent)
        sent_score = [0.0] * length

        if length == 0:
            return char_score, 0.0

        # 遍历每一个黑名单词
        for gram in self.black_list:
            gram_len = len(gram)
            if gram_len == 0:
                continue

            # 在句子中查找所有出现位置
            start = 0
            while True:
                pos = sent.find(gram, start)
                if pos == -1:
                    break
                # 命中后，对应字符位置打 -1 分
                for k in range(gram_len):
                    if pos + k < length:
                        sent_score[pos + k] = min(-1, sent_score[pos + k] )
                start = pos + 1  # 继续往后找

        # 映射到全局 char_score
        for i in range(length):
            global_idx = i + idx
            # 如果出现问题，打印debug信息
            if global_idx >= len(char_score):
                print("[DEBUG] char_score out of range!")
                print(f"  sent: {sent!r}")
                print(f"  idx: {idx}")
                print(f"  sent_score: {sent_score}")
                print(f"  black_list: {self.black_list}")
                print(f"  char_score(len={len(char_score)}): {char_score}")
                print(f"  global_idx: {global_idx}, i: {i}")
                return char_score, sum(sent_score) / length + 1
            # 否则赋值
            char_score[global_idx] += sent_score[i]
        # 返回结果
        return char_score, sum(sent_score) / length + 1
    
    def extract_model_output(self, model_output: str, refer_message: str):
        """
        从模型输出的文本中提取出需要的信息，包括：
        【卡片标题】：{title}
        【要点】：{question}
        【回答】：{answer}
        参考信息格式：
        pagetitle__bidword

        返回：
            title, question, answer, pagetitle, bidword,
            title_start, question_start, answer_start
        """
        # 解析卡片信息
        title, question, answer = '', '', ''
        title_start, question_start, answer_start = -1, -1, -1

        offset = 0
        for line in model_output.split('\n'):
            line_stripped = line.strip()
            if line_stripped.startswith('【卡片标题】：'):
                title = line_stripped.replace('【卡片标题】：', '').strip()
                pos = line.find(title)
                title_start = offset + pos if pos != -1 and title else -1
            elif line_stripped.startswith('【要点】：'):
                question = line_stripped.replace('【要点】：', '').strip()
                pos = line.find(question)
                question_start = offset + pos if pos != -1 and question else -1
            elif line_stripped.startswith('【回答】：'):
                answer = line_stripped.replace('【回答】：', '').strip()
                pos = line.find(answer)
                answer_start = offset + pos if pos != -1 and answer else -1
            offset += len(line) + 1  # +1 是换行符

        # 解析参考信息
        pagetitle, bidword = '', ''
        if '__' in refer_message:
            pagetitle, bidword = refer_message.split('__', 1)
            if "__pbsim" in bidword:
                bidword = bidword.split('__')[0].strip()

        return title, question, answer, pagetitle, bidword, title_start, question_start, answer_start

    def parse_relevance_output(self, text: str) -> dict:
        """
        解析 judge model 的输出文本，提取【相关性】分数（int）
        
        输出格式要求：
        【相关性】：具体分数，如（0分、1分、2分、3分、4分、5分）
        【理由】：具体理由
        """
        score = 0
        if not text:
            return score
        # 提取相关性分数
        score_match = re.search(r"【相关性】\s*[:：]?\s*([0-5])分", text)
        if score_match:
            try:
                score = int(score_match.group(1))
            except ValueError:
                score = 0
        return score

    def model_output_to_judge_input_qwen(self, title, question, answer, bidword) -> str:
        """
        输入给judge model判断相关性
        """
        reward_input = (PROMPT_BASE,
            f"广告主买词：{bidword}",
            f"广告内容：",
            f"【卡片标题】：{title}",
            f"【要点】：{question}",
            f"【回答】：{answer}",
            f"#输出：")
        return "\n".join(reward_input)
        
    def model_output_to_reward_input_qwen(self, title, question, answer, pagetitle, bidword) -> str:
        """
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

    def calc_overall_reward(self, rule_score, model_score, relevant_score):
        return self.reword_weight * model_score + self.rule_weight * rule_score + self.relevant_weight * relevant_score

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
        judge_model_inputs = []
        all_rule_score_dicts = []
        titles = []
        questions = []
        answers = []
        char_rewards = []
        response_str_list=[]
        for i in range(batch_size):
            valid_response_ids = response_ids[i][: response_length[i]]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            response_str_list.append(response_str)
            char_reward = [0.0]*len(response_str)
            log(f"first,len of str:{len(response_str)}, and len of char_reward:{len(char_reward)}")
            char_rewards.append(char_reward)

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
                title, question, answer, pagetitle, bidword, title_start, question_start, answer_start = self.extract_model_output(response_str, refer_msg)
                reward_input = self.model_output_to_reward_input_qwen(title, question, answer, pagetitle, bidword)
                judge_input = self.model_output_to_judge_input_qwen(title, question, answer, bidword)
                reward_model_inputs.append(reward_input)
                judge_model_inputs.append(judge_input)
                titles.append((title,title_start))
                questions.append((question,question_start))
                answers.append((answer,answer_start))
            else:
                reward_model_inputs.append(None)
                judge_model_inputs.append(None)
                titles.append(None)
                questions.append(None)
                answers.append(None)
            log(f"sample {i}: rule_score={rule_score}, response_len={int(response_length[i])}")

        # Step 2: batch 模型推理
        model_scores = [0.0] * batch_size
        ctr_predict = [0.0] * batch_size
        cvr_predict = [0.0] * batch_size
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
                        for idx, score in zip(batch_indices, batch_ctr_scores):
                            ctr_predict[idx] = score
                        for idx, score in zip(batch_indices, batch_ctcvr_scores):
                            # cvr = ctcvr/ctr
                            if ctr_predict[idx] != 0:
                                cvr_predict[idx] = score / ctr_predict[idx]
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

        # Step 3: 相关性审核模型
        relevent_scores = [0.0] * batch_size
        # 在 compute_reward 里
        if self.judge_model is not None:
            # 只收集非 None 的样本
            valid_indices_and_prompts = [(i, p) for i, p in enumerate(judge_model_inputs) if p is not None]
            if valid_indices_and_prompts:
                batch_indices = [idx for idx, _ in valid_indices_and_prompts]
                batch_prompts = [p for _, p in valid_indices_and_prompts]
                log(f"judge forward batch: size={len(batch_prompts)}")

                outputs = self.judge_model.generate(batch_prompts, self.vllm_sampling_params)
                # outputs: List[RequestOutput]，每个output.outputs[0].text是生成文本

                for idx, output in zip(batch_indices, outputs):
                    t = output.outputs[0].text.strip()
                    # print(f"batch_indices:{idx} and output is {t}")
                    label = self.parse_relevance_output(t)  # 解析为具体分数
                    try:
                        relevent_scores[idx] = float(label) / 5
                    except Exception as e:
                        print(f"sample {idx}: judge_output={t}, label parse failed: {e}")
                        relevent_scores[idx] = 0.0
                    log(f"sample {idx}: judge_output={t}, label={label}")
            else:
                log("no valid judge inputs, skip relevance scoring")

        # Step 4: 多样性奖励
        # 当前 char_rewards大小为(bts, len(response_str))
        diversity_score_dicts = []  # 每个样本的多样性得分字典
        if batch_size > 1:
            # 提取n_grams
            high_freq_ngrams_title = self.calc_high_freq_multi_ngram_penalty(titles)
            print("the patterns in title will be punished" , high_freq_ngrams_title)
            # high_freq_ngrams_question = self.calc_high_freq_multi_ngram_penalty(questions)
            # print("the patterns in question will be punished" , high_freq_ngrams_question)
            high_freq_ngrams_answer = self.calc_high_freq_multi_ngram_penalty(answers)
            print("the patterns in answer will be punished" , high_freq_ngrams_answer)
            # 进行token级别的reward赋值
            for i in range(batch_size):
                # 下面的diversity的逻辑是扣分逻辑，所以这里先普+1
                log(f"before process, len char_reward:{len(char_rewards[i])}, len str:{len(response_str_list[i])}")
                char_rewards[i] = [a + 1.0 for a in char_rewards[i]]
                log(f"after add, len char_reward:{len(char_rewards[i])}, len str:{len(response_str_list[i])}")
                # 这里对句子级别的损失和char级别的损失都进行了返回
                char_rewards[i], diversity_title = self.get_ngram_char_level_reward(titles[i], high_freq_ngrams_title, char_rewards[i])
                # char_rewards[i], diversity_question = self.get_ngram_char_level_reward(questions[i], high_freq_ngrams_question, char_rewards[i])
                char_rewards[i], diversity_answer = self.get_ngram_char_level_reward(answers[i], high_freq_ngrams_answer, char_rewards[i])
                log(f"after diversity process, len char_reward:{len(char_rewards[i])}, len str:{len(response_str_list[i])}")
                diversity_dict = {
                    "[Diversity]title_diversity_score":  diversity_title,
                    "[Diversity]question_diversity_score": 1.0,
                    "[Diversity]answer_diversity_score": diversity_answer,
                }
                diversity_score_dicts.append(diversity_dict)
        else:
            # batch_size == 1 时，定义多样性得分为0.0
            for _ in range(batch_size):
                diversity_dict = {
                    "[Diversity]title_diversity_score": 0.0,
                    "[Diversity]question_diversity_score": 0.0,
                    "[Diversity]answer_diversity_score": 0.0,
                }
                diversity_score_dicts.append(diversity_dict)

        # Step 5: 风控黑名单
        black_score_dicts = [] 
        if self.black_list:
            if batch_size > 1:
                # 进行token级别的reward赋值
                for i in range(batch_size):
                    # 下面的black list的逻辑是扣分逻辑，所以这里先普+1
                    char_rewards[i] = [a + 1.0 for a in char_rewards[i]]
                    # 这里对句子级别的损失和char级别的损失都进行了返回
                    char_rewards[i], black_title = self.get_black_list_char_level_reward(titles[i], char_rewards[i])
                    char_rewards[i], black_question = self.get_black_list_char_level_reward(questions[i], char_rewards[i])
                    char_rewards[i], black_answer = self.get_black_list_char_level_reward(answers[i], char_rewards[i])
                    log(f"after black list, len char_reward:{len(char_rewards[i])}, len str:{len(response_str_list[i])}")
                    black_dict = {
                        "[Black]title_black_score":  black_title,
                        "[Black]question_black_score": black_question,
                        "[Black]answer_black_score": black_answer,
                    }
                    black_score_dicts.append(black_dict)
            else:
                # batch_size == 1 时，定义多样性得分为0.0
                for _ in range(batch_size):
                    black_score_dicts = {
                        "[Black]title_black_score":  0.0,
                        "[Black]question_black_score": 0.0,
                        "[Black]answer_black_score": 0.0,
                    }
                    black_score_dicts.append(black_score_dicts)

        # Step 6: 计算 overall reward
        for i in range(batch_size):
            diversity_score = (diversity_score_dicts[i]["[Diversity]answer_diversity_score"] + diversity_score_dicts[i]["[Diversity]question_diversity_score"] + diversity_score_dicts[i]["[Diversity]title_diversity_score"]) / 3
            black_score = (black_score_dicts[i]["[Black]answer_black_score"] + black_score_dicts[i]["[Black]question_black_score"] + black_score_dicts[i]["[Black]title_black_score"]) / 3
            overall = self.calc_overall_reward(rule_scores[i], model_scores[i], relevent_scores[i])
            # 重新取对应的valid_response_ids
            valid_response_ids = response_ids[i][: response_length[i]]
            token_rewards = self.map_char_rewards_to_tokens(
                valid_response_ids, self.tokenizer, response_str_list[i], char_rewards[i])
            token_rewards_tensor = torch.tensor(token_rewards, dtype=reward_tensor.dtype, device=reward_tensor.device)
            reward_tensor[i, :len(token_rewards)] = token_rewards_tensor + overall

            reward_metrics["overall"].append(overall)
            reward_metrics["rule_score"].append(rule_scores[i])
            reward_metrics["model_score"].append(model_scores[i])
            reward_metrics["relevant_score"].append(relevent_scores[i])
            reward_metrics["diversity_score"].append(diversity_score)
            reward_metrics["black_score"].append(black_score)
            reward_metrics["ctr_predict"].append(ctr_predict[i])
            reward_metrics["cvr_predict"].append(cvr_predict[i])
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