import torch
from torch import nn
from loss.said import saidloss


SCENARIO_NAMES = ("dt-st", "dt-lt", "nt-st", "nt-lt", "ad-st", "ad-lt")
TASK_GROUPS = {
    "dt": ("dt-st", "dt-lt"),
    "nt": ("nt-st", "nt-lt"),
    "ad": ("ad-st", "ad-lt"),
}
TIME_GROUPS = {
    "st": ("dt-st", "nt-st", "ad-st"),
    "lt": ("dt-lt", "nt-lt", "ad-lt"),
}


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

    def _compute_scenario_losses(self, y, pids, cids, mids):
        return {
            scenario: saidloss(
                y[index],
                pids,
                cids,
                mids,
                self.invalid_same_pid_mask,
                scenario,
                self.said,
                self.hdw,
            )
            for index, scenario in enumerate(SCENARIO_NAMES)
        }

    def _hdw_weight(self, *probability_groups):
        merged = [group for group in probability_groups if group.numel() > 0]
        if not merged:
            return torch.zeros((), device=self.invalid_same_pid_mask.device)
        return (1 - self._safe_mean(torch.cat(merged, dim=0))).clamp_min(1e-6).sqrt()

    def forward(self, y, pids, cids, mids):
        losses = self._compute_scenario_losses(y, pids, cids, mids)
        if not self.hdw:
            return sum(losses.values())

        probabilities = {scenario: (-loss).exp() for scenario, loss in losses.items()}
        task_weights = {
            task_name: self._hdw_weight(*(probabilities[name] for name in scenario_names))
            for task_name, scenario_names in TASK_GROUPS.items()
        }
        time_weights = {
            time_name: self._hdw_weight(*(probabilities[name] for name in scenario_names))
            for time_name, scenario_names in TIME_GROUPS.items()
        }
        w_tm = torch.stack(tuple(task_weights.values())).max().clamp_min(1e-6)
        w_ti = torch.stack(tuple(time_weights.values())).max().clamp_min(1e-6)

        weighted_losses = {}
        for scenario_name, loss_value in losses.items():
            task_name, time_name = scenario_name.split('-')
            weighted_losses[scenario_name] = (
                (task_weights[task_name] / w_tm) *
                (time_weights[time_name] / w_ti) *
                loss_value
            )

        batch_size = float(pids.shape[0])
        loss = sum(
            value.sum() if value.ndim > 0 else value
            for value in weighted_losses.values()
        )
        return loss / batch_size
