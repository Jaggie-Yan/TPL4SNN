import torch
from torch.nn import functional as F
from util.progressive_loss import *

def kd_loss_pertime(outputs_kd, teacher_outputs, T):
    distillation_loss_pertime =  F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction="sum",
                log_target=True,
            ) * (T * T) / outputs_kd.numel()
    return distillation_loss_pertime


class DistillationLoss(torch.nn.Module):

    def __init__(
        self,
        base_criterion: torch.nn.Module,
        teacher_model: torch.nn.Module,
        distillation_type: str,
        alpha: float,
        tau: float,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ["none", "soft", "hard"]
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = Progressive_loss(outputs, labels, self.base_criterion)
        if self.distillation_type == "none":
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "When knowledge distillation is enabled, the model is "
                "expected to return a Tuple[Tensor, Tensor] with the output of the "
                "class_token and the dist_token"
            )
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            T = self.tau
            distillation_loss = (
                F.kl_div(
                    F.log_softmax(outputs_kd.mean(0) / T, dim=1),
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction="sum",
                    log_target=True,
                )
                * (T * T)
                / outputs_kd.mean(0).numel()
            )
        elif self.distillation_type == "hard":
            distillation_loss = F.cross_entropy(
                outputs_kd.mean(0), teacher_outputs.argmax(dim=1)
            )

        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss
