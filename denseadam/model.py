"""
与 RLlib FCNet 兼容的分析模型，供 functorch 计算梯度方差使用
"""

import gym.spaces
import torch
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.typing import Dict, List, ModelConfigDict, TensorType
from typing import Tuple


class UnifiedFullyConnectedNetwork(FullyConnectedNetwork):
    """与 FullyConnectedNetwork 兼容，forward 使用 obs_flat，便于 functorch 调用"""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        # 兼容 obs_flat 或 obs（RLlib 不同场景）
        obs = input_dict.get("obs_flat", input_dict.get("obs"))
        obs = torch.as_tensor(obs, dtype=torch.float32)
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        if self._value_branch_separate:
            value = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            value = self._value_branch(self._features).squeeze(1)
        return logits, [value]
