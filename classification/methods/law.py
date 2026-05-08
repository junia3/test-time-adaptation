"""
Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation.

Adapted from:
https://github.com/junia3/LayerwiseTTA

Paper:
Layer-wise Auto-Weighting for Non-Stationary Test-Time Adaptation, WACV 2024.
"""

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from augmentations.transforms_cotta import get_tta_transforms
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class LAW(TTAMethod):
    """Layer-wise Auto-Weighting."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        if self.optimizer is None:
            raise ValueError("LAW requires at least one trainable parameter.")

        self.base_lr = self.optimizer.param_groups[0]["lr"]
        self.tau = cfg.LAW.TAU
        self.eps = 1e-8

        # Important compatibility change:
        # The current test-time-adaptation repo expects img_size here,
        # while the original LayerwiseTTA repo used dataset_name.
        self.transforms = get_tta_transforms(self.img_size)

        self.grad_weight = defaultdict(float)
        self.trainable_dict = {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt the model on a batch of test data."""

        imgs = x[0]

        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(imgs)
        logits_aug = self.model(self.transforms(imgs))

        pseudo_labels = logits.detach().argmax(dim=1)
        fisher_loss = F.cross_entropy(logits, pseudo_labels)
        fisher_loss.backward(retain_graph=True)

        min_weight = float("inf")
        max_weight = float("-inf")

        for name, param in self.trainable_dict.items():
            if param.grad is None:
                continue

            # The original LAW code accumulates squared gradients and later uses
            # sqrt(mean(.)); accumulating the scalar mean gives the same statistic
            # for the final layer-wise learning-rate weight and saves memory.
            self.grad_weight[name] += param.grad.detach().pow(2).mean().item()

            value = self.grad_weight[name] ** 0.5
            min_weight = min(min_weight, value)
            max_weight = max(max_weight, value)

        param_groups = []
        for name, value_sq in self.grad_weight.items():
            value = value_sq ** 0.5
            lr_weight = (value - min_weight) / (max_weight - min_weight + self.eps)

            param_groups.append(
                {
                    "params": [self.trainable_dict[name]],
                    "lr": self.base_lr * (lr_weight ** self.tau),
                }
            )

        if not param_groups:
            self.optimizer.zero_grad(set_to_none=True)
            return logits

        self.optimizer = self._build_law_optimizer(param_groups)
        self.optimizer.zero_grad(set_to_none=True)

        loss = softmax_entropy(logits)
        loss += 0.01 * logits.shape[1] * consistency(logits, logits_aug)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        return logits

    def _build_law_optimizer(self, param_groups):
        optim_method = self.cfg.OPTIM.METHOD

        if optim_method == "Adam":
            return torch.optim.Adam(
                param_groups,
                betas=(self.cfg.OPTIM.BETA, 0.999),
                weight_decay=self.cfg.OPTIM.WD,
            )

        if optim_method == "AdamW":
            return torch.optim.AdamW(
                param_groups,
                betas=(self.cfg.OPTIM.BETA, 0.999),
                weight_decay=self.cfg.OPTIM.WD,
            )

        if optim_method == "SGD":
            return torch.optim.SGD(
                param_groups,
                momentum=self.cfg.OPTIM.MOMENTUM,
                dampening=self.cfg.OPTIM.DAMPENING,
                weight_decay=self.cfg.OPTIM.WD,
                nesterov=self.cfg.OPTIM.NESTEROV,
            )

        raise NotImplementedError(f"Unknown optimizer: {optim_method}")

    def reset(self):
        """Reset model and optimizer.

        LAW rebuilds the optimizer with one parameter group per trainable tensor
        during adaptation. The base reset method would try to load the original
        optimizer state into this changed optimizer structure, which can cause a
        parameter-group mismatch. Therefore, rebuild the base optimizer first.
        """
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state.")

        for model, model_state in zip(self.models, self.model_states):
            model.load_state_dict(model_state, strict=True)

        self.optimizer = self.setup_optimizer()
        self.optimizer.load_state_dict(self.optimizer_state)

        self.grad_weight = defaultdict(float)

        if hasattr(self, "input_buffer"):
            self.input_buffer = None
        if hasattr(self, "pointer"):
            self.pointer.zero_()
        if hasattr(self, "performed_updates"):
            self.performed_updates = 0

    def collect_params(self):
        """Collect trainable parameters for LAW."""
        params = []
        names = []

        for module_name, module in self.model.named_modules():
            if isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                    nn.Conv2d,
                ),
            ):
                for param_name, param in module.named_parameters():
                    if param_name in ["weight", "bias"] and param.requires_grad:
                        params.append(param)
                        names.append(f"{module_name}.{param_name}")

        return params, names

    def configure_model(self):
        """Configure model for LAW."""
        self.model.eval()
        self.model.requires_grad_(False)

        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.requires_grad_(True)
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

            elif isinstance(
                module,
                (
                    nn.BatchNorm1d,
                    nn.LayerNorm,
                    nn.GroupNorm,
                    nn.Conv2d,
                ),
            ):
                module.train()
                module.requires_grad_(True)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()


@torch.jit.script
def consistency(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()