"""
DenseAdam PPO 训练器：工厂函数、模型恢复与继续训练
"""

import os

import cloudpickle
import torch
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

from denseadam.optimizer import GradientNormSquaredOptimizer
from denseadam.policy import DenseAdamPPOTorchPolicy


def get_trainer_class(config):
    """创建使用 DenseAdam 的 CustomPPOTrainer 类"""
    print("\n" + "=" * 50)
    print("[get_trainer_class] 开始创建 DenseAdam PPO 训练器类...")

    custom_config = DEFAULT_CONFIG.copy()
    default_opt = {
        "threshold": 100.0,
        "percentile": 90.0,
        "ema_alpha": 0.5,
        "base_optimizer": torch.optim.Adam,
        "log_file": None,
    }
    user_opt = config.get("custom_optimizer", {})
    custom_config["custom_optimizer"] = {**default_opt, **user_opt}

    # 确保 custom_optimizer 中的 base_optimizer 为类而非字符串
    opt_cfg = custom_config["custom_optimizer"]
    if isinstance(opt_cfg.get("base_optimizer"), str):
        import importlib

        mod_name, cls_name = opt_cfg["base_optimizer"].rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        opt_cfg["base_optimizer"] = getattr(mod, cls_name)

    class DenseAdamPPOTrainer(PPOTrainer):
        @classmethod
        def get_default_config(cls):
            return custom_config

        @classmethod
        def get_default_policy_class(cls, config):
            return DenseAdamPPOTorchPolicy

    print("[get_trainer_class] 自定义训练器类创建完成")
    print("=" * 50 + "\n")
    return DenseAdamPPOTrainer


def restore_model(checkpoint_path: str, config: dict):
    """
    从 PPO checkpoint 恢复模型，用于 DenseAdam 继续训练

    Args:
        checkpoint_path: checkpoint 文件路径
        config: 配置字典（需包含 env, num_gpus, custom_optimizer 等）

    Returns:
        trainer 实例，失败时返回 None
    """
    try:
        with open(checkpoint_path, "rb") as f:
            checkpoint_data = cloudpickle.load(f)

        worker_bytes = checkpoint_data["worker"]
        worker = cloudpickle.loads(worker_bytes)
        train_exec_impl = checkpoint_data["train_exec_impl"]

        policy_weights = worker["state"]["default_policy"]["weights"]

        trainer_class = get_trainer_class(config)
        trainer_config = {
            "env": config["env"],
            "num_gpus": config["num_gpus"],
            "num_workers": config["num_workers"],
            "num_envs_per_worker": config["num_envs_per_worker"],
            "train_batch_size": config["train_batch_size"],
            "batch_mode": config["batch_mode"],
            "framework": config["framework"],
            "ignore_worker_failures": True,
            "custom_optimizer": config.get("custom_optimizer", {}),
        }
        trainer = trainer_class(config=trainer_config)

        policy = trainer.get_policy("default_policy")

        if not isinstance(policy, DenseAdamPPOTorchPolicy):
            raise ValueError(
                f"策略加载错误！预期 DenseAdamPPOTorchPolicy，"
                f"实际 {policy.__class__.__name__}"
            )

        model_state_dict = {}
        for param_name, param_value in policy_weights.items():
            model_state_dict[param_name] = torch.tensor(
                param_value, dtype=torch.float32
            )

        policy.model.load_state_dict(model_state_dict)
        print(f"模型权重加载成功: {checkpoint_path}")

        info = train_exec_impl["info"]
        if "learner" in info:
            learner_stats = info["learner"]["default_policy"]["learner_stats"]
            current_lr = learner_stats["cur_lr"]
            for param_group in policy.optimizer.param_groups:
                param_group["lr"] = current_lr
            current_kl_coeff = learner_stats["cur_kl_coeff"]
            trainer.workers.local_worker().foreach_policy(
                lambda p, _: setattr(p, "kl_coeff", current_kl_coeff)
            )
            print(f"已恢复学习率: {current_lr}, KL系数: {current_kl_coeff}")

        counters = train_exec_impl["counters"]
        trainer._iteration = counters["num_steps_trained"] // max(
            counters.get("num_steps_trained_this_iter", 1), 1
        )
        print(f"已恢复迭代次数: {trainer._iteration}")

        return trainer

    except Exception as e:
        print(f"模型加载过程中发生错误: {e}")
        return None


def continue_training(trainer, num_iterations: int, save_dir: str):
    """
    继续训练指定迭代次数，并在每轮后保存 checkpoint

    Args:
        trainer: 已恢复的 DenseAdamPPOTrainer
        num_iterations: 继续训练的迭代次数
        save_dir: 保存目录
    """
    iteration_initial = trainer._iteration
    print(f"当前迭代次数: {iteration_initial}, 将继续训练 {num_iterations} 次")

    os.makedirs(save_dir, exist_ok=True)
    print(f"继续训练模型，保存路径: {save_dir}")

    for i in range(num_iterations):
        result = trainer.train()

        policy = trainer.get_policy()
        optimizer = policy.optimizer
        if isinstance(optimizer, GradientNormSquaredOptimizer):
            optimizer.update_threshold_based_on_epoch()
            stats = optimizer.get_stats()
            print(f"优化器统计：{stats}")

        print(
            f"Iteration {iteration_initial + 1 + i}: "
            f"reward = {result.get('episode_reward_mean', 'N/A')}"
        )
        print(f"平均奖励: {result.get('episode_reward_mean', 'N/A')}")
        print(f"最大奖励: {result.get('episode_reward_max', 'N/A')}")
        print(f"最小奖励: {result.get('episode_reward_min', 'N/A')}")

        trainer.save(save_dir)
        print(f"模型已保存到: {save_dir}/checkpoint-{iteration_initial + 1 + i}")
