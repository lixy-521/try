"""
使用 functorch 计算 mini-batch 梯度方差的工具函数
"""

import torch
from functorch import grad, make_functional, vmap
from ray.rllib.policy.sample_batch import SampleBatch

from denseadam.model import UnifiedFullyConnectedNetwork


def compute_minibatch_grad_variance(
    policy,
    model: UnifiedFullyConnectedNetwork,
    train_batch: SampleBatch,
    device: str = "cuda:0",
) -> tuple:
    """
    计算当前 mini-batch 的梯度方差（用于 DenseAdam 更新前判断）

    Args:
        policy: PPOTorchPolicy 实例
        model: 分析用模型（需与 policy.model 结构一致）
        train_batch: 当前 mini-batch 的 SampleBatch
        device: 计算设备

    Returns:
        (grad_variance, all_elem_var): 梯度方差标量、全元素方差
    """
    model.to(device)
    fmodel, params = make_functional(model)

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.detach().float().to(device)
        return torch.as_tensor(x, dtype=torch.float32).to(device)

    states = _to_tensor(train_batch["obs"])
    actions = _to_tensor(train_batch["actions"])
    action_dist_inputs = _to_tensor(train_batch["action_dist_inputs"])
    action_logps = _to_tensor(train_batch["action_logp"])
    advantages = _to_tensor(train_batch["advantages"])
    vf_preds = _to_tensor(train_batch["vf_preds"])
    value_targets = _to_tensor(train_batch["value_targets"])

    def compute_loss(params, s, a, adi, alp, adv, vfp, vt):
        s = s.unsqueeze(0)
        a = a.unsqueeze(0)
        adi = adi.unsqueeze(0)
        alp = alp.unsqueeze(0)
        adv = adv.unsqueeze(0)
        vfp = vfp.unsqueeze(0)
        vt = vt.unsqueeze(0)

        input_dict = {"obs": s, "obs_flat": s}
        logits, value_fn_out = fmodel(
            params,
            input_dict=input_dict,
            state=[torch.tensor([0.0], device=s.device)],
            seq_lens=torch.tensor([1], device=s.device),
        )
        mean, log_std = torch.chunk(logits, 2, dim=1)
        curr_action_dist = torch.distributions.normal.Normal(
            mean, torch.exp(log_std), validate_args=False
        )
        prev_mean, prev_log_std = torch.chunk(adi, 2, dim=1)
        prev_action_dist = torch.distributions.normal.Normal(
            prev_mean, torch.exp(prev_log_std), validate_args=False
        )

        logp_ratio = torch.exp(curr_action_dist.log_prob(a).sum(-1) - alp)
        surrogate_loss = torch.min(
            adv * logp_ratio,
            adv
            * torch.clamp(
                logp_ratio,
                1 - policy.config["clip_param"],
                1 + policy.config["clip_param"],
            ),
        )
        vf_loss = (
            torch.pow(value_fn_out[0] - vt, 2.0)
            if policy.config["use_critic"]
            else torch.tensor(0.0, device=s.device)
        )
        total_loss = torch.mean(
            -surrogate_loss + policy.config["vf_loss_coeff"] * vf_loss
        )
        return total_loss

    grads = vmap(grad(compute_loss), (None, 0, 0, 0, 0, 0, 0, 0))(
        params,
        states,
        actions,
        action_dist_inputs,
        action_logps,
        advantages,
        vf_preds,
        value_targets,
    )
    grad_flat = torch.cat(
        [g.detach().view(len(states), -1) for g in grads], dim=1
    )
    grad_variance = grad_flat.var(dim=0, unbiased=True).sum().item()
    all_elem_var = torch.var(grad_flat).item()
    return grad_variance, all_elem_var
