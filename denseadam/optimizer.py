"""
DenseAdam 优化器 (GradientNormSquaredOptimizer)
基于梯度方差阈值的自定义优化器，支持动态 EMA 阈值更新
"""

import os
import numpy as np
import torch
from torch.optim import Optimizer


class GradientNormSquaredOptimizer(Optimizer):
    """
    基于梯度范数平方的自定义优化器，支持动态调整阈值
    当梯度方差小于阈值时执行更新，否则跳过
    新增指数滑动平均(EMA)机制平滑阈值更新
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        threshold=100.0,
        percentile=90,
        ema_alpha=0.5,
        optimizer_class=torch.optim.Adam,
        log_file=None,
        **kwargs,
    ):
        print("\n[GradientNormSquaredOptimizer] 初始化自定义优化器...")

        if threshold <= 0.0:
            raise ValueError(f"阈值必须为正值，得到 {threshold}")

        if not (0 <= percentile <= 100):
            raise ValueError(f"分位数必须在0-100之间，得到 {percentile}")

        if not (0 <= ema_alpha <= 1):
            raise ValueError(f"EMA平滑系数必须在(0,1)之间，得到 {ema_alpha}")

        defaults = dict(
            lr=lr,
            threshold=threshold,
            percentile=percentile,
            ema_alpha=ema_alpha,
            optimizer_class=optimizer_class,
            **kwargs,
        )
        super(GradientNormSquaredOptimizer, self).__init__(params, defaults)

        self.current_threshold = threshold

        param_list = list(params)
        if not param_list:
            print("[GradientNormSquaredOptimizer] 警告：传入的参数列表为空！")
        else:
            print(
                f"[GradientNormSquaredOptimizer] 成功接收 {len(param_list)} 组可训练参数"
            )

        self.base_optimizer = optimizer_class(params, lr=lr, **kwargs)

        self.total_steps = 0
        self.skipped_steps = 0
        self.epoch_grad_norms = []
        self.all_grad_norms = []
        self.epoch_grad_vars = []
        self.all_grad_vars = []

        self.log_file = log_file
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(log_file, "w") as f:
                f.write(
                    f"步数,梯度方差,全元素方差,梯度二范数平方,阈值,是否更新\n"
                )
            print(f"[GradientNormSquaredOptimizer] 日志文件已创建: {log_file}")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        grad_norm_squared = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("GradientNormSquaredOptimizer不支持稀疏梯度")
                grad_norm_squared += torch.sum(grad * grad).item()

        self.epoch_grad_norms.append(grad_norm_squared)
        self.all_grad_norms.append(grad_norm_squared)
        self.total_steps += 1

        grad_variance = getattr(self, "current_batch_grad_variance", None)
        all_elem_var = getattr(self, "current_batch_all_elem_var", None)
        batch_size = getattr(self, "current_batch_size", 128)

        if grad_variance is not None:
            self.epoch_grad_vars.append(grad_variance)
            self.all_grad_vars.append(grad_variance)
        else:
            print("[GradientNormSquaredOptimizer] 当前batch梯度方差未设置，跳过打印")
            self.skipped_steps += 1
            return loss

        # 平均梯度二范数平方（除以 batch_size^2 归一化）
        grad_norm_squared_avg = (
            grad_norm_squared / (batch_size * batch_size)
            if batch_size > 0
            else grad_norm_squared
        )

        # 每步输出三项信息
        all_elem_var_str = f"{all_elem_var:.6f}" if all_elem_var is not None else "N/A"
        print(
            f"[GradientNormSquaredOptimizer] 第{self.total_steps}步: "
            f"梯度方差(协方差迹/参数数)={grad_variance:.6f}, "
            f"全元素方差(torch.var)={all_elem_var_str}, "
            f"梯度二范数平方={grad_norm_squared:.4f}(/n²={grad_norm_squared_avg:.4f})"
        )

        threshold = self.current_threshold
        update_performed = False
        if grad_variance < threshold:
            self.base_optimizer.step()
            update_performed = True
            status_msg = (
                f"第{self.total_steps}步: 梯度方差({grad_variance:.4f}) < "
                f"阈值({threshold:.4f}), 执行更新"
            )
        else:
            self.skipped_steps += 1
            status_msg = (
                f"第{self.total_steps}步: 梯度方差({grad_variance:.4f}) >= "
                f"阈值({threshold:.4f}), 跳过更新"
            )

        print(status_msg)

        if self.log_file:
            all_elem_var_log = (
                f"{all_elem_var:.6f}" if all_elem_var is not None else ""
            )
            with open(self.log_file, "a") as f:
                f.write(
                    f"{self.total_steps},{grad_variance:.6f},"
                    f"{all_elem_var_log},{grad_norm_squared:.4f},"
                    f"{threshold:.4f},{update_performed}\n"
                )

        return loss

    def update_threshold_based_on_epoch(self):
        """基于当前 epoch 的梯度方差更新阈值（EMA）"""
        if not self.epoch_grad_vars:
            print(
                "[GradientNormSquaredOptimizer] 警告："
                "当前epoch没有记录的梯度方差，无法更新阈值"
            )
            return

        percentile = self.param_groups[0]["percentile"]
        ema_alpha = self.param_groups[0]["ema_alpha"]
        current_quantile = np.percentile(self.epoch_grad_vars, percentile)

        self.current_threshold = (
            ema_alpha * current_quantile + (1 - ema_alpha) * self.current_threshold
        )

        print(
            f"\n[GradientNormSquaredOptimizer] 阈值更新: "
            f"当前{percentile}%分位数={current_quantile:.4f}, "
            f"EMA系数={ema_alpha}, 新阈值={self.current_threshold:.4f}"
        )
        print(
            f"[GradientNormSquaredOptimizer] 当前epoch梯度方差统计: "
            f"最小值={min(self.epoch_grad_vars):.4f}, "
            f"最大值={max(self.epoch_grad_vars):.4f}, "
            f"平均值={np.mean(self.epoch_grad_vars):.4f}"
        )

        self.epoch_grad_vars = []
        return self.current_threshold

    def get_stats(self):
        skip_ratio = (
            self.skipped_steps / self.total_steps if self.total_steps > 0 else 0
        )
        return {
            "total_steps": self.total_steps,
            "skipped_steps": self.skipped_steps,
            "skip_ratio": skip_ratio,
            "current_threshold": self.current_threshold,
            "percentile": self.param_groups[0]["percentile"],
            "ema_alpha": self.param_groups[0]["ema_alpha"],
        }
