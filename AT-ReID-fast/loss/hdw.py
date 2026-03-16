import torch
from torch import nn
from loss.said import HEAD_SPECS, compute_head_losses


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
    def __init__(self, cid2pid, said=False, hdw=False):
        super(saidloss_hdw, self).__init__()
        self.said = said
        self.hdw = hdw
        cid_to_pid = torch.tensor([cid2pid[i] for i in range(len(cid2pid))], dtype=torch.long)
        self.register_buffer('cid_to_pid', cid_to_pid, persistent=False)
        invalid_same_pid_mask = cid_to_pid.unsqueeze(0).eq(cid_to_pid.unsqueeze(1))
        invalid_same_pid_mask.fill_diagonal_(False)
        self.register_buffer('invalid_same_pid_mask', invalid_same_pid_mask, persistent=False)

    @staticmethod
    def _safe_mean(loss):
        if loss.ndim == 0:
            return loss
        if loss.numel() == 0:
            return loss.new_zeros(())
        return loss.mean()

    def _hdw_weight(self, *probability_groups):
        merged = [group for group in probability_groups if group.numel() > 0]
        if not merged:
            return torch.zeros((), device=self.invalid_same_pid_mask.device)
        return (1 - self._safe_mean(torch.cat(merged, dim=0))).clamp_min(1e-6).sqrt()

    def forward(self, y, pids, cids, mids):
        losses = compute_head_losses(
            y,
            pids,
            cids,
            mids,
            self.invalid_same_pid_mask,
            said=self.said,
            hdw=self.hdw,
        )
        if not self.hdw:
            return sum(losses)

        probabilities = tuple((-loss.detach()).exp() for loss in losses)
        task_weights = tuple(
            self._hdw_weight(*(probabilities[index] for index in group_indices))
            for group_indices in TASK_GROUP_INDICES
        )
        time_weights = tuple(
            self._hdw_weight(*(probabilities[index] for index in group_indices))
            for group_indices in TIME_GROUP_INDICES
        )
        w_tm = torch.stack(task_weights).max().clamp_min(1e-6)
        w_ti = torch.stack(time_weights).max().clamp_min(1e-6)

        weighted_losses = tuple(
            (task_weights[HEAD_TASK_INDEX[index]] / w_tm) *
            (time_weights[HEAD_TIME_INDEX[index]] / w_ti) *
            loss_value
            for index, loss_value in enumerate(losses)
        )

        batch_size = float(pids.shape[0])
        loss = sum(
            value.sum() if value.ndim > 0 else value
            for value in weighted_losses
        )
        return loss / batch_size
