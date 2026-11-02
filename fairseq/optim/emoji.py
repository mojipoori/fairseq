# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.distributed as dist
import torch.optim

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class FairseqEmojiConfig(FairseqDataclass):

    emoji_betas: Any = field(
        default=(0.9, 0.999),
        metadata={"help": "betas for Emoji optimizer"},
    )

    emoji_eps: float = field(
        default=1e-8,
        metadata={"help": "epsilon for Emoji optimizer"},
    )

    weight_decay: float = field(
        default=0.0,
        metadata={"help": "weight decay"},
    )

    weight_decouple: bool = field(
        default=True,
        metadata={"help": "use decoupled weight decay"},
    )

    amsgrad: bool = field(
        default=False,
        metadata={"help": "use the AMSGrad variant of Emoji"},
    )

    tpu: bool = II("common.tpu")

    lr: List[float] = II("optimization.lr")


@register_optimizer("emoji", dataclass=FairseqEmojiConfig)
class FairseqEmoji(FairseqOptimizer):

    """
    Emoji optimizer for fairseq.

    This implementation wraps the original Emoji optimizer
    without changing its optimization algorithm.
    """

    def __init__(self, cfg: FairseqEmojiConfig, params):

        super().__init__(cfg)

        self._optimizer = Emoji(
            params,
            **self.optimizer_config,
        )

    @property
    def optimizer_config(self):

        """
        Return optimizer arguments used to construct Emoji.

        These values can override optimizer arguments stored in
        checkpoints when resuming training.
        """

        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,

            "betas": eval(self.cfg.emoji_betas)
            if isinstance(self.cfg.emoji_betas, str)
            else OmegaConf.to_container(self.cfg.emoji_betas),

            "eps": self.cfg.emoji_eps,

            "weight_decay": self.cfg.weight_decay,

            "weight_decouple": self.cfg.weight_decouple,

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

            value["exp_var"] /= total_gpus

            value["jerk"] /= total_gpus

            dist.all_reduce(
                value["exp_avg"],
                op=dist.ReduceOp.SUM,
            )

            dist.all_reduce(
                value["exp_var"],
                op=dist.ReduceOp.SUM,
            )

            dist.all_reduce(
                value["jerk"],
                op=dist.ReduceOp.SUM,
            )


class Emoji(torch.optim.Optimizer):

    """pure default mojic

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 weight_decouple=True, amsgrad=False):

        if not 0.0 <= lr:

            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0.0 <= eps:

            raise ValueError(f"Invalid epsilon value: {eps}")

        if not 0.0 <= betas[0] < 1.0:

            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")

        if not 0.0 <= betas[1] < 1.0:

            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        if not 0.0 <= weight_decay:

            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        weight_decouple=weight_decouple, amsgrad=amsgrad,
                        )

        super(Emoji, self).__init__(params, defaults)

    def __setstate__(self, state):

        super(Emoji, self).__setstate__(state)

    def step(self, closure=None):

        """Performs a single optimization step.

        Arguments:

            closure (callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        loss = None

        if closure is not None:

            with torch.enable_grad():

                loss = closure()

        for group in self.param_groups:

            for p in group['params']:

                if p.grad is None:

                    continue

                grad = p.grad.data

                if grad.is_sparse:

                    raise RuntimeError(

                        "MojiC does not support sparse gradients, "

                        "please consider SparseAdam instead"

                    )

                state = self.state[p]

                # State initialization

                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['exp_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['jerk'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    if group['amsgrad']:

                        state['max_exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']

                exp_var = state['exp_var']

                jerk = state['jerk']

                beta1, beta2 = group['betas']

                if group['amsgrad']:

                    max_exp_avg_var = state['max_exp_avg_var']

                state['step'] += 1

                current_step = state['step']

                # Apply weight decay (before rectify, as in MojiC original)

                if group['weight_decay'] != 0 and not group['weight_decouple']:

                    grad = grad.add(p.data, alpha=group['weight_decay'])

                if group['weight_decay'] != 0 and group['weight_decouple']:

                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update first moment (m)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Compute variance (s)

                d = grad - exp_avg

                exp_var.mul_(beta1).add_(d, alpha=1 - beta1)

                # Compute jerk (j)

                jerk.mul_(beta2).addcmul_(d - exp_var, d - exp_var, value=1 - beta2)

                # Bias correction

                bias_correction1 = 1 - beta1 ** current_step

                bias_correction2 = 1 - beta2 ** current_step

                m_hat = exp_avg / bias_correction1

                j_hat = jerk / bias_correction2

                if group['amsgrad']:

                    torch.max(max_exp_avg_var, jerk, out=max_exp_avg_var)

                    denom = max_exp_avg_var.sqrt().add_(group['eps'])

                else:

                    denom = j_hat.sqrt().add_(group['eps'])

                # Standard MojiC update

                p.data.addcdiv_(m_hat, denom, value=-group['lr'])

        return loss
    
    