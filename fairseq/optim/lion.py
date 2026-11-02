# Copyright 2023 Google Research. All Rights Reserved.
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
# ==============================================================================

"""PyTorch implementation of the Lion optimizer."""

import torch

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, List

from torch.optim.optimizer import Optimizer

from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from omegaconf import II, OmegaConf


@dataclass
class FairseqLionConfig(FairseqDataclass):
    lion_betas: Any = field(
        default=(0.9, 0.99),
        metadata={"help": "betas for Lion optimizer"},
    )

    weight_decay: float = field(
        default=0.0,
        metadata={"help": "weight decay"},
    )

    tpu: bool = II("common.tpu")

    lr: List[float] = II("optimization.lr")


@register_optimizer("lion", dataclass=FairseqLionConfig)
class FairseqLion(FairseqOptimizer):
    """
    Lion optimizer for fairseq.
    """

    def __init__(self, cfg: FairseqLionConfig, params):
        super().__init__(cfg)

        self._optimizer = Lion(
            params,
            **self.optimizer_config,
        )

    @property
    def optimizer_config(self):
        """
        Return optimizer arguments used to construct Lion.
        """

        return {
            "lr": self.cfg.lr[0]
            if isinstance(self.cfg.lr, Collection)
            else self.cfg.lr,

            "betas": (
                eval(self.cfg.lion_betas)
                if isinstance(self.cfg.lion_betas, str)
                else OmegaConf.to_container(self.cfg.lion_betas)
            ),

            "weight_decay": self.cfg.weight_decay,
        }


class Lion(Optimizer):

    r"""Implements Lion algorithm."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):

        if not 0.0 <= lr:
            raise ValueError('Invalid learning rate: {}'.format(lr))

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 0: {}'.format(betas[0])
            )

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                'Invalid beta parameter at index 1: {}'.format(betas[1])
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:

                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                beta1, beta2 = group['betas']

                # Weight update
                update = exp_avg * beta1 + grad * (1 - beta1)

                p.add_(update.sign_(), alpha=-group['lr'])

                # Decay the momentum running average coefficient
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss