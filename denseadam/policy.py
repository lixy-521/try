"""
自定义 PPO 策略：集成 DenseAdam 优化器，在 learn_on_loaded_batch 中计算梯度方差
"""

import torch
from ray.rllib.agents.ppo import PPOTorchPolicy
from ray.rllib.models.catalog import ModelCatalog

from denseadam.model import UnifiedFullyConnectedNetwork
from denseadam.optimizer import GradientNormSquaredOptimizer
from denseadam.variance import compute_minibatch_grad_variance


class DenseAdamPPOTorchPolicy(PPOTorchPolicy):
    """使用 DenseAdam 的 PPO 策略"""

    def __init__(self, observation_space, action_space, config):
        print("\n" + "=" * 50)
        print("[DenseAdamPPOTorchPolicy] 开始初始化策略...")
        print(
            f"[DenseAdamPPOTorchPolicy] 观测空间: {observation_space}, "
            f"动作空间: {action_space}"
        )

        super().__init__(observation_space, action_space, config)
        print("[DenseAdamPPOTorchPolicy] 父类初始化完成")

        _, logit_dim = ModelCatalog.get_action_dist(
            self.action_space, self.config["model"], framework=self.framework
        )
        self.analysis_model = UnifiedFullyConnectedNetwork(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            name="MiniBatchAnalysisModel",
        )
        print("[DenseAdamPPOTorchPolicy] 分析模型初始化完成")

        if not hasattr(self, "model"):
            raise ValueError("父类未初始化model属性！")

        model_params = list(self.model.parameters())
        if not model_params:
            raise ValueError("模型没有可训练参数！")
        print(f"[DenseAdamPPOTorchPolicy] 模型参数数量: {len(model_params)} 组")

        self._optimizer_initialized = False
        self.after_init()

    def after_init(self):
        self._model_params = list(self.model.parameters())

        if not self._model_params:
            print("[DenseAdamPPOTorchPolicy] 警告：模型参数为空！")
        else:
            optimizer_config = self.config.get("custom_optimizer", {})
            threshold = optimizer_config.get("threshold", 100.0)
            percentile = optimizer_config.get("percentile", 90)
            ema_alpha = optimizer_config.get("ema_alpha", 0.5)
            base_optimizer = optimizer_config.get("base_optimizer", torch.optim.Adam)
            log_file = optimizer_config.get("log_file", None)

            self.optimizer = GradientNormSquaredOptimizer(
                self._model_params,
                lr=self.config["lr"],
                threshold=threshold,
                percentile=percentile,
                ema_alpha=ema_alpha,
                optimizer_class=base_optimizer,
                log_file=log_file,
            )
            self._optimizers = [self.optimizer]
            self._optimizer_initialized = True
            print("[DenseAdamPPOTorchPolicy] 优化器初始化完成")

    def learn_on_loaded_batch(self, offset: int = 0, buffer_index: int = 0):
        if not self._loaded_batches[buffer_index]:
            raise ValueError("必须先调用 load_batch_into_buffer()！")

        cuda_device = self.config.get("cuda_device", "0")
        device_str = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"

        try:
            device_batch_size = self.config.get(
                "sgd_minibatch_size", self.config["train_batch_size"]
            ) // len(self.devices)

            buffer_batches = self._loaded_batches[buffer_index]
            if device_batch_size >= sum(len(b) for b in buffer_batches):
                current_mini_batches = buffer_batches
            else:
                current_mini_batches = [
                    b[offset : offset + device_batch_size] for b in buffer_batches
                ]

            current_batch = current_mini_batches[0]
            self.analysis_model.load_state_dict(self.model.state_dict())

            current_batch_var, all_elem_var = compute_minibatch_grad_variance(
                policy=self,
                model=self.analysis_model,
                train_batch=current_batch,
                device=device_str,
            )

            param_count = sum(p.numel() for p in self.model.parameters())
            current_batch_var_normalized = current_batch_var / param_count

            if hasattr(self, "optimizer"):
                self.optimizer.current_batch_grad_variance = (
                    current_batch_var_normalized
                )
                self.optimizer.current_batch_all_elem_var = all_elem_var
                self.optimizer.current_batch_size = len(current_batch)

            if self.config.get("verbose", False):
                print(
                    f"[方差计算完成] 当前mini-batch方差(除以参数数): "
                    f"{current_batch_var_normalized:.6f}, "
                    f"所有元素的方差: {all_elem_var:.6f}"
                )

        except Exception as e:
            print(f"[错误] 梯度方差计算失败: {str(e)}")

        result = super().learn_on_loaded_batch(offset=offset, buffer_index=buffer_index)
        return result
