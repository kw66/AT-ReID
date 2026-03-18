import torch
from torch import nn

from loss.said import HEAD_SCENARIOS, HEAD_SPECS, compute_head_losses


TASK_NAMES = ("dt", "nt", "ad")
TIME_NAMES = ("st", "lt")
TASK_GROUP_INDICES = tuple(
    tuple(index for index, spec in enumerate(HEAD_SPECS) if spec["task"] == task_name)
    for task_name in TASK_NAMES
)
TIME_GROUP_INDICES = tuple(
    tuple(index for index, spec in enumerate(HEAD_SPECS) if spec["time"] == time_name)
    for time_name in TIME_NAMES
)
HEAD_TASK_INDEX = tuple(TASK_NAMES.index(spec["task"]) for spec in HEAD_SPECS)
HEAD_TIME_INDEX = tuple(TIME_NAMES.index(spec["time"]) for spec in HEAD_SPECS)


class saidloss_hdw(nn.Module):
    """Fixed fast HDW implementation used by AT-ReID-fast.

    The historical ablation interfaces have been removed. When HDW is enabled,
    the loss always uses the single retained configuration:

    - mode = legacy-group
    - group source = score-cat
    - score = norm-nll
    - score level = sample
    - class count = effective
    - normalize = max
    - group reduce = mean
    - loss reduce = batch-sum
    """

    def __init__(self, cid2pid, said=False, hdw=False):
        super().__init__()
        self.said = said
        self.hdw = hdw
        cid_to_pid = torch.tensor([cid2pid[i] for i in range(len(cid2pid))], dtype=torch.long)
        invalid_same_pid_mask = cid_to_pid.unsqueeze(0).eq(cid_to_pid.unsqueeze(1))
        invalid_same_pid_mask.fill_diagonal_(False)
        self.register_buffer("invalid_same_pid_mask", invalid_same_pid_mask, persistent=False)

    @staticmethod
    def _safe_mean(loss):
        if loss.ndim == 0:
            return loss
        if loss.numel() == 0:
            return loss.new_zeros(())
        return loss.mean()

    @staticmethod
    def _scenario_active(loss):
        return bool(loss.ndim == 0 or loss.numel() > 0)

    @staticmethod
    def _stabilize_class_counts(class_counts):
        return class_counts.float().clamp_min(2.0)

    def _sample_norm_nll(self, losses, class_counts):
        if losses.numel() == 0:
            return losses.new_zeros((0,))
        losses = losses.float()
        class_counts = self._stabilize_class_counts(class_counts.to(dtype=losses.dtype, device=losses.device))
        return losses / class_counts.log().clamp_min(1e-6)

    def _group_active_mask(self, losses, group_indices):
        device = losses[0].device if losses else self.invalid_same_pid_mask.device
        values = [
            any(self._scenario_active(losses[index]) for index in indices)
            for indices in group_indices
        ]
        return torch.tensor(values, device=device, dtype=torch.bool)

    def _reduce_group_tensor(self, tensors):
        merged = [tensor.reshape(-1).float() for tensor in tensors if tensor.numel() > 0]
        if not merged:
            return torch.zeros((), device=self.invalid_same_pid_mask.device)
        return torch.cat(merged, dim=0).mean()

    def _normalize_active_scores(self, scores, active_mask):
        weights = torch.zeros_like(scores)
        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return weights
        active_scores = scores.index_select(0, active_indices).clamp_min(1e-6)
        normalized = active_scores / active_scores.max().clamp_min(1e-6)
        weights.index_copy_(0, active_indices, normalized)
        return weights

    def _compute_fixed_hdw_weights(self, losses, effective_class_counts):
        raw_scores = []
        class_count_means = []
        sample_score_values = []
        active_mask_values = []
        for loss, effective_class_count in zip(losses, effective_class_counts):
            is_active = self._scenario_active(loss)
            active_mask_values.append(is_active)
            if effective_class_count.ndim == 0:
                class_count_mean = effective_class_count.float()
            elif effective_class_count.numel() == 0:
                class_count_mean = loss.new_zeros(())
            else:
                class_count_mean = effective_class_count.float().mean()
            class_count_means.append(class_count_mean)
            if not is_active:
                raw_scores.append(loss.new_zeros(()))
                sample_score_values.append(loss.new_zeros((0,)))
                continue
            detached_loss = loss.detach().float()
            detached_counts = effective_class_count.detach().float()
            sample_scores = self._sample_norm_nll(detached_loss, detached_counts)
            raw_scores.append(sample_scores.mean())
            sample_score_values.append(sample_scores)

        raw_score_tensor = torch.stack(raw_scores)
        smoothed_score_tensor = raw_score_tensor.detach().clone()
        class_count_mean_tensor = torch.stack(class_count_means)
        scenario_active_mask = torch.tensor(active_mask_values, device=raw_score_tensor.device, dtype=torch.bool)
        task_active_mask = self._group_active_mask(losses, TASK_GROUP_INDICES)
        time_active_mask = self._group_active_mask(losses, TIME_GROUP_INDICES)

        task_scores = torch.stack([
            self._reduce_group_tensor(tuple(sample_score_values[index] for index in group_indices))
            for group_indices in TASK_GROUP_INDICES
        ])
        time_scores = torch.stack([
            self._reduce_group_tensor(tuple(sample_score_values[index] for index in group_indices))
            for group_indices in TIME_GROUP_INDICES
        ])
        task_weights = self._normalize_active_scores(task_scores, task_active_mask)
        time_weights = self._normalize_active_scores(time_scores, time_active_mask)
        scenario_weights = torch.stack([
            task_weights[HEAD_TASK_INDEX[index]] * time_weights[HEAD_TIME_INDEX[index]]
            for index in range(len(HEAD_SCENARIOS))
        ])
        return (
            scenario_weights,
            raw_score_tensor,
            smoothed_score_tensor,
            class_count_mean_tensor,
            task_weights,
            time_weights,
            scenario_active_mask,
        )

    def _build_details(
        self,
        losses,
        scenario_weights,
        effective_weights,
        base_losses,
        loss_contribs,
        batch_size,
        *,
        raw_scores=None,
        smoothed_scores=None,
        class_count_means=None,
        task_group_weights=None,
        time_group_weights=None,
    ):
        scenario_mean_losses = {
            name: float(self._safe_mean(loss).item())
            for name, loss in zip(HEAD_SCENARIOS, losses)
        }
        scenario_sample_counts = {
            name: int(loss.numel()) if loss.ndim > 0 else int(batch_size)
            for name, loss in zip(HEAD_SCENARIOS, losses)
        }
        details = {
            "weight_mode": "legacy-group-fast" if self.hdw else "none",
            "scenario_mean_losses": scenario_mean_losses,
            "scenario_weights": {
                name: float(weight.item())
                for name, weight in zip(HEAD_SCENARIOS, scenario_weights)
            },
            "scenario_effective_weights": {
                name: float(weight.item())
                for name, weight in zip(HEAD_SCENARIOS, effective_weights)
            },
            "scenario_base_losses": {
                name: float(loss_value.item())
                for name, loss_value in zip(HEAD_SCENARIOS, base_losses)
            },
            "scenario_loss_contribs": {
                name: float(loss_value.item())
                for name, loss_value in zip(HEAD_SCENARIOS, loss_contribs)
            },
            "scenario_sample_counts": scenario_sample_counts,
        }
        if raw_scores is not None:
            details["scenario_raw_scores"] = {
                name: float(score.item())
                for name, score in zip(HEAD_SCENARIOS, raw_scores)
            }
        if smoothed_scores is not None:
            details["scenario_smoothed_scores"] = {
                name: float(score.item())
                for name, score in zip(HEAD_SCENARIOS, smoothed_scores)
            }
        if class_count_means is not None:
            details["scenario_class_count_means"] = {
                name: float(score.item())
                for name, score in zip(HEAD_SCENARIOS, class_count_means)
            }
        if task_group_weights is not None:
            details["task_group_weights"] = {
                name: float(weight.item())
                for name, weight in zip(TASK_NAMES, task_group_weights)
            }
        if time_group_weights is not None:
            details["time_group_weights"] = {
                name: float(weight.item())
                for name, weight in zip(TIME_NAMES, time_group_weights)
            }
        return details

    def forward(self, y, pids, cids, mids, return_details=False):
        losses, effective_class_counts, _ = compute_head_losses(
            y,
            pids,
            cids,
            mids,
            self.invalid_same_pid_mask,
            said=self.said,
            hdw=True,
            return_class_counts=True,
        )

        mean_losses = tuple(self._safe_mean(loss) for loss in losses)
        if not self.hdw:
            scenario_weight_tensor = torch.ones(len(HEAD_SCENARIOS), device=mean_losses[0].device, dtype=mean_losses[0].dtype)
            base_loss_tensor = torch.stack(mean_losses)
            loss_contrib_tensor = base_loss_tensor
            details = self._build_details(
                losses,
                tuple(scenario_weight_tensor[index] for index in range(len(HEAD_SCENARIOS))),
                tuple(scenario_weight_tensor[index] for index in range(len(HEAD_SCENARIOS))),
                tuple(base_loss_tensor[index] for index in range(len(HEAD_SCENARIOS))),
                tuple(loss_contrib_tensor[index] for index in range(len(HEAD_SCENARIOS))),
                pids.shape[0],
            )
            total_loss = loss_contrib_tensor.sum()
            if return_details:
                return total_loss, details
            return total_loss

        batch_size = float(pids.shape[0])
        (
            scenario_weight_tensor,
            raw_scores,
            smoothed_scores,
            class_count_means,
            task_group_weights,
            time_group_weights,
            _scenario_active_mask,
        ) = self._compute_fixed_hdw_weights(losses, effective_class_counts)
        base_loss_tensor = torch.stack([
            (loss_value.sum() / batch_size) if loss_value.ndim > 0 else loss_value
            for loss_value in losses
        ])
        effective_weight_tensor = torch.stack([
            scenario_weight_tensor[index] * (float(loss.numel()) / batch_size)
            for index, loss in enumerate(losses)
        ])
        loss_contrib_tensor = scenario_weight_tensor.to(dtype=base_loss_tensor.dtype) * base_loss_tensor
        total_loss = loss_contrib_tensor.sum()
        details = self._build_details(
            losses,
            tuple(scenario_weight_tensor[index] for index in range(len(HEAD_SCENARIOS))),
            tuple(effective_weight_tensor[index] for index in range(len(HEAD_SCENARIOS))),
            tuple(base_loss_tensor[index] for index in range(len(HEAD_SCENARIOS))),
            tuple(loss_contrib_tensor[index] for index in range(len(HEAD_SCENARIOS))),
            batch_size,
            raw_scores=raw_scores,
            smoothed_scores=smoothed_scores,
            class_count_means=class_count_means,
            task_group_weights=task_group_weights,
            time_group_weights=time_group_weights,
        )
        if return_details:
            return total_loss, details
        return total_loss
