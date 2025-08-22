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
"""
Reward config
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    # basic
    reward_type: str = "batch"
    reward_function: Optional[str] = None
    reward_function_kwargs: dict = field(default_factory=dict)
    skip_special_tokens: bool = True
    num_cpus: int = 1
    # reward weight
    rule_weight: float = 0.5
    diversity_weight: float = 0.5
    relevant_weight: float = 0.5
    # reward model
    reward_model_path: Optional[str] = None
    reward_model_weight: float = 0.5
    model_score_type: str = "ctcvr"
    model_q_process: str = "log"
    reward_model_batch_size: int = 1024
    # judge model
    judge_model_path: Optional[str] = None
    judge_model_batch_size: int = 256
    # diversity
    n_gram_low_bound: int = 2
    n_gram_up_bound: int = 5
    n_gram_threshold: float = 0.15
    ngram_penalty: float = 0.1
    # black list
    black_list_path: Optional[str] = None
    # below are auto keys
    reward_function_name: Optional[str] = field(default=None, init=False)

    def post_init(self):
        if self.reward_function is not None:  # support custom reward function, e.g., ./math.py:main
            if ":" not in self.reward_function:
                self.reward_function_name = "main"
            else:
                self.reward_function, self.reward_function_name = self.reward_function.rsplit(":", maxsplit=1)

            if os.path.exists(self.reward_function):  # ray job uses absolute path
                self.reward_function = os.path.abspath(self.reward_function)
            else:
                print(f"Reward function {self.reward_function} not found.")
                self.reward_function = None
