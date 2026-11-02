# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import math

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
class FairseqAdaBeliefConfig(FairseqDataclass):

    adabelief_betas: Any = field(
        default=(0.9, 0.999),
        metadata={"help": "betas for AdaBelief optimizer"},
    )

    adabelief_eps: float = field(
        default=1e-8,
        metadata={"help": "epsilon for AdaBelief optimizer"},
    )

    weight_decay: float = field(
        default=0.0,
        metadata={"help": "weight decay"},
    )

    weight_decouple: bool = field(
        default=True,
        metadata={"help": "use decoupled weight decay"},
    )

    fixed_decay: bool = field(
        default=False,
        metadata={"help": "use fixed weight decay"},
    )

    rectify: bool = field(
        default=False,
        metadata={"help": "use rectification"},
    )

    degenerated_to_sgd: bool = field(
        default=False,
        metadata={"help": "use SGD when variance is high"},
    )

    amsgrad: bool = field(
        default=False,
        metadata={"help": "use AMSGrad"},
    )

    tpu: bool = II("common.tpu")

    lr: List[float] = II("optimization.lr")


@register_optimizer("adabelief", dataclass=FairseqAdaBeliefConfig)
class FairseqAdaBelief(FairseqOptimizer):

    """
    AdaBelief optimizer for fairseq.
    """

    def __init__(self, cfg: FairseqAdaBeliefConfig, params):

        super().__init__(cfg)

        self._optimizer = AdaBelief(
            params,
            **self.optimizer_config,
        )

    @property
    def optimizer_config(self):

        """
        Return optimizer arguments used to construct AdaBelief.

        These values can override optimizer arguments stored in
        checkpoints when resuming training.
        """

        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,

            "betas": eval(self.cfg.adabelief_betas)
            if isinstance(self.cfg.adabelief_betas, str)
            else OmegaConf.to_container(self.cfg.adabelief_betas),

            "eps": self.cfg.adabelief_eps,

            "weight_decay": self.cfg.weight_decay,

            "amsgrad": self.cfg.amsgrad,

            "weight_decouple": self.cfg.weight_decouple,

            "fixed_decay": self.cfg.fixed_decay,

            "rectify": self.cfg.rectify,

            "degenerated_to_sgd": self.cfg.degenerated_to_sgd,
        }

    def average_params(self):

        """
        Reduce optimizer states for BMUF distributed training.
        """

        state_dict = self.optimizer.state_dict()

        total_gpus = float(dist.get_world_size())

        for _, value in state_dict["state"].items():

            value["exp_avg"] /= total_gpus

            value["exp_avg_var"] /= total_gpus

            dist.all_reduce(
                value["exp_avg"],
                op=dist.ReduceOp.SUM,
            )

            dist.all_reduce(
                value["exp_avg_var"],
                op=dist.ReduceOp.SUM,
            )

            if "max_exp_avg_var" in value:

                value["max_exp_avg_var"] /= total_gpus

                dist.all_reduce(
                    value["max_exp_avg_var"],
                    op=dist.ReduceOp.SUM,
                )


class AdaBelief(torch.optim.Optimizer):

    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch

    Arguments:

        params (iterable): iterable of parameters to optimize or dicts defining

            parameter groups

        lr (float, optional): learning rate (default: 1e-3)

        betas (Tuple[float, float], optional): coefficients used for computing

            running averages of gradient and its square (default: (0.9, 0.999))

        eps (float, optional): term added to the denominator to improve

            numerical stability (default: 1e-16)

        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

        amsgrad (boolean, optional): whether to use the AMSGrad variant of this

            algorithm from the paper `On the Convergence of Adam and Beyond`

        weight_decouple (boolean, optional): ( default: True) If set as True, then

            the optimizer uses decoupled weight decay as in AdamW

        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple

            is set as True.

        rectify (boolean, optional): (default: False) If set as True, then perform the rectified

            update similar to RAdam

        degenerated_to_sgd (boolean, optional) (default: False) If set as True, then perform SGD update

            when variance of gradient is high

    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020

    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,

                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=False,

                 degenerated_to_sgd=False):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd

        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):

            for param in params:

                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):

                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            buffer=[[None, None, None] for _ in range(10)]
        )

        super(AdaBelief, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd

        self.weight_decouple = weight_decouple

        self.rectify = rectify

        self.fixed_decay = fixed_decay

    def __setstate__(self, state):

        super(AdaBelief, self).__setstate__(state)

        for group in self.param_groups:

            group.setdefault('amsgrad', False)

    def reset(self):

        for group in self.param_groups:

            for p in group['params']:

                state = self.state[p]

                amsgrad = group['amsgrad']

                state['step'] = 0

                state['exp_avg'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format
                )

                state['exp_avg_var'] = torch.zeros_like(
                    p.data,
                    memory_format=torch.preserve_format
                )

                if amsgrad:

                    state['max_exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format
                    )

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

                        'AdaBelief does not support sparse gradients, please consider SparseAdam instead'

                    )

                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2 = group['betas']

                if len(state) == 0:

                    state['step'] = 0

                    state['exp_avg'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format
                    )

                    state['exp_avg_var'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format
                    )

                    if amsgrad:

                        state['max_exp_avg_var'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format
                        )

                exp_avg = state['exp_avg']

                exp_avg_var = state['exp_avg_var']

                state['step'] += 1

                bias_correction1 = 1 - beta1 ** state['step']

                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_residual = grad - exp_avg

                exp_avg_var.mul_(beta2).addcmul_(
                    grad_residual,
                    grad_residual,
                    value=1 - beta2
                )

                if amsgrad:

                    max_exp_avg_var = state['max_exp_avg_var']

                    torch.max(
                        max_exp_avg_var,
                        exp_avg_var,
                        out=max_exp_avg_var
                    )

                    denom = (
                        max_exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                else:

                    denom = (
                        exp_avg_var.add_(group['eps']).sqrt()
                        / math.sqrt(bias_correction2)
                    ).add_(group['eps'])

                if self.weight_decouple:

                    if not self.fixed_decay:

                        p.data.mul_(
                            1.0 - group['lr'] * group['weight_decay']
                        )

                    else:

                        p.data.mul_(
                            1.0 - group['weight_decay']
                        )

                else:

                    if group['weight_decay'] != 0:

                        grad.add_(
                            p.data,
                            alpha=group['weight_decay']
                        )

                if not self.rectify:

                    step_size = group['lr'] / bias_correction1

                    p.data.addcdiv_(
                        exp_avg,
                        denom,
                        value=-step_size
                    )

                else:

                    buffered = group['buffer'][int(state['step'] % 10)]

                    if state['step'] == buffered[0]:

                        N_sma, step_size = buffered[1], buffered[2]

                    else:

                        buffered[0] = state['step']

                        beta2_t = beta2 ** state['step']

                        N_sma_max = 2 / (1 - beta2) - 1

                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                        buffered[1] = N_sma

                        if N_sma >= 5:

                            step_size = math.sqrt(

                                (1 - beta2_t)
                                * (N_sma - 4)
                                / (N_sma_max - 4)
                                * (N_sma - 2)
                                / N_sma
                                * N_sma_max
                                / (N_sma_max - 2)

                            ) / (1 - beta1 ** state['step'])

                        elif self.degenerated_to_sgd:

                            step_size = 1.0 / (1 - beta1 ** state['step'])

                        else:

                            step_size = -1

                        buffered[2] = step_size

                    if N_sma >= 5:

                        denom = exp_avg_var.sqrt().add_(group['eps'])

                        p.data.addcdiv_(
                            exp_avg,
                            denom,
                            value=-step_size * group['lr']
                        )

                    elif step_size > 0:

                        p.data.add_(
                            exp_avg,
                            alpha=-step_size * group['lr']
                        )

        return loss