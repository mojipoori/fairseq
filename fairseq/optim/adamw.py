# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.distributed as dist
import torch.optim

from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class FairseqAdamWConfig(FairseqDataclass):
    adam_betas: Any = field(
        default=(0.9, 0.999),
        metadata={"help": "betas for AdamW optimizer"},
    )

    adam_eps: float = field(
        default=1e-8,
        metadata={"help": "epsilon for AdamW optimizer"},
    )

    weight_decay: float = field(
        default=1e-2,
        metadata={"help": "decoupled weight decay"},
    )

    amsgrad: bool = field(
        default=False,
        metadata={"help": "use the AMSGrad variant of AdamW"},
    )

    tpu: bool = II("common.tpu")

    lr: List[float] = II("optimization.lr")


@register_optimizer("adamw", dataclass=FairseqAdamWConfig)
class FairseqAdamW(FairseqOptimizer):
    """
    AdamW optimizer for fairseq.

    This implementation follows the decoupled weight decay formulation
    of AdamW.

    The implementation is intentionally independent from
    fairseq.optim.adam.Adam so that Adam and AdamW can be compared
    independently.
    """

    def __init__(self, cfg: FairseqAdamWConfig, params):
        super().__init__(cfg)

        self._optimizer = AdamW(
            params,
            **self.optimizer_config,
        )

    @property
    def optimizer_config(self):
        """
        Return optimizer arguments used to construct AdamW.

        These values can override optimizer arguments stored in
        checkpoints when resuming training.
        """

        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,
            "betas": eval(self.cfg.adam_betas)
            if isinstance(self.cfg.adam_betas, str)
            else OmegaConf.to_container(self.cfg.adam_betas),
            "eps": self.cfg.adam_eps,
            "weight_decay": self.cfg.weight_decay,
            "amsgrad": self.cfg.amsgrad,
        }

    def average_params(self):
        """
        Reduce optimizer states for BMUF distributed training.
        """

        state_dict = self.optimizer.state_dict()

        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():
            value["exp_avg"] /= total_gpus
            value["exp_avg_sq"] /= total_gpus

            dist.all_reduce(
                value["exp_avg"],
                op=dist.ReduceOp.SUM,
            )

            dist.all_reduce(
                value["exp_avg_sq"],
                op=dist.ReduceOp.SUM,
            )


class AdamW(torch.optim.Optimizer):
    r"""
    Implements the AdamW algorithm.

    AdamW uses decoupled weight decay. The parameter decay is applied
    directly to the parameters and is not included in the first or
    second moment estimates.

    The implementation follows the formulation:

        theta_t = theta_{t-1}
                  - lr * weight_decay * theta_{t-1}

        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t

        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2

        theta_t = theta_t
                  - lr / (1 - beta1^t)
                  * m_t
                  / (
                      sqrt(v_t / (1 - beta2^t))
                      + eps
                    )

    This implementation is independent of fairseq.optim.adam.Adam.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError(
                "Invalid learning rate: {}".format(lr)
            )

        if not 0.0 <= eps:
            raise ValueError(
                "Invalid epsilon value: {}".format(eps)
            )

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]
                )
            )

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]
                )
            )

        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(
                    weight_decay
                )
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

        super(AdamW, self).__init__(
            params,
            defaults,
        )

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return True

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)

        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """
        Performs a single optimization step.

        The order of operations is:

        1. Decoupled weight decay
        2. First moment update
        3. Second moment update
        4. Bias correction
        5. Parameter update
        """

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group["params"]:

                if p.grad is None:
                    continue

                # Get gradient.
                grad = p.grad.data

                # Convert reduced precision gradients to FP32.
                if grad.dtype in {
                    torch.float16,
                    torch.bfloat16,
                }:
                    grad = grad.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )

                amsgrad = group["amsgrad"]

                # Keep parameters in FP32 for the optimizer computation
                # when using FP16/BF16 parameters.
                p_data_fp32 = p.data

                if p.data.dtype in {
                    torch.float16,
                    torch.bfloat16,
                }:
                    p_data_fp32 = p_data_fp32.float()

                state = self.state[p]

                # ---------------------------------------------------------
                # State initialization
                # ---------------------------------------------------------

                if len(state) == 0:

                    state["step"] = 0

                    # Exponential moving average of gradients.
                    state["exp_avg"] = torch.zeros_like(
                        p_data_fp32
                    )

                    # Exponential moving average of squared gradients.
                    state["exp_avg_sq"] = torch.zeros_like(
                        p_data_fp32
                    )

                    if amsgrad:
                        # Maximum of the second moment estimates.
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p_data_fp32
                        )

                else:

                    state["exp_avg"] = state["exp_avg"].to(
                        p_data_fp32
                    )

                    state["exp_avg_sq"] = state["exp_avg_sq"].to(
                        p_data_fp32
                    )

                    if amsgrad:
                        state["max_exp_avg_sq"] = (
                            state["max_exp_avg_sq"].to(
                                p_data_fp32
                            )
                        )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]

                beta1, beta2 = group["betas"]

                # ---------------------------------------------------------
                # Increment optimization step
                # ---------------------------------------------------------

                state["step"] += 1

                # ---------------------------------------------------------
                # Decoupled weight decay
                #
                # theta <- theta * (1 - lr * weight_decay)
                #
                # IMPORTANT:
                # This is independent of the gradient/moment calculation.
                # ---------------------------------------------------------

                if group["weight_decay"] != 0:

                    p_data_fp32.mul_(
                        1
                        - group["lr"]
                        * group["weight_decay"]
                    )

                # ---------------------------------------------------------
                # First moment
                #
                # m_t =
                # beta1 * m_{t-1}
                # + (1 - beta1) * g_t
                # ---------------------------------------------------------

                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # ---------------------------------------------------------
                # Second moment
                #
                # v_t =
                # beta2 * v_{t-1}
                # + (1 - beta2) * g_t^2
                # ---------------------------------------------------------

                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # ---------------------------------------------------------
                # Bias corrections
                # ---------------------------------------------------------

                bias_correction1 = (
                    1
                    - beta1 ** state["step"]
                )

                bias_correction2 = (
                    1
                    - beta2 ** state["step"]
                )

                # ---------------------------------------------------------
                # Denominator
                #
                # sqrt(
                #     v_t / (1 - beta2^t)
                # ) + eps
                # ---------------------------------------------------------

                if amsgrad:

                    torch.max(
                        max_exp_avg_sq,
                        exp_avg_sq,
                        out=max_exp_avg_sq,
                    )

                    denom = (
                        max_exp_avg_sq.sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(
                        group["eps"]
                    )

                else:

                    denom = (
                        exp_avg_sq.sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(
                        group["eps"]
                    )

                # ---------------------------------------------------------
                # Step size
                #
                # lr / (1 - beta1^t)
                # ---------------------------------------------------------

                step_size = (
                    group["lr"]
                    / bias_correction1
                )

                # ---------------------------------------------------------
                # Parameter update
                #
                # theta <- theta
                #         - step_size * m_t / denom
                # ---------------------------------------------------------

                p_data_fp32.addcdiv_(
                    exp_avg,
                    denom,
                    value=-step_size,
                )

                # ---------------------------------------------------------
                # Copy FP32 parameters back to FP16/BF16 parameters.
                # ---------------------------------------------------------

                if p.data.dtype in {
                    torch.float16,
                    torch.bfloat16,
                }:

                    p.data.copy_(
                        p_data_fp32
                    )

        return loss