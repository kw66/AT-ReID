import torch
import torch.nn.functional as F


SCENARIO_TO_MODALITY = {
    "dt-st": 1,
    "dt-lt": 1,
    "nt-st": 2,
    "nt-lt": 2,
}

SHORT_TERM_SCENARIOS = {"dt-st", "nt-st", "ad-st"}
LONG_TERM_SCENARIOS = {"dt-lt", "nt-lt", "ad-lt"}


def _gather_nll(log_probs, targets):
    return -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)


def _select_scenario_samples(y, pids, cids, mids, scenario: str):
    if scenario not in SCENARIO_TO_MODALITY:
        return y, pids, cids, mids
    mask = mids == SCENARIO_TO_MODALITY[scenario]
    return y[mask], pids[mask], cids[mask], mids[mask]


def _empty_loss(y, *, hdw: bool):
    return y.new_zeros((0,)) if hdw else y.new_zeros(())


def _long_term_loss(logits, pids, *, hdw: bool):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    losses = _gather_nll(log_probs, pids)
    return losses if hdw else losses.mean()


def _short_term_loss(logits, cids, invalid_same_pid_mask, *, hdw: bool):
    invalid_mask = invalid_same_pid_mask.index_select(0, cids)
    masked_logits = logits.float().masked_fill(invalid_mask, -1e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    losses = _gather_nll(log_probs, cids)
    return losses if hdw else losses.mean()


def saidloss(y, pids, cids, mids, invalid_same_pid_mask, scenario, said=False, hdw=False):
    if not said:
        scenario = 'ad-lt'

    y, pids, cids, mids = _select_scenario_samples(y, pids, cids, mids, scenario)
    if y.numel() == 0:
        return _empty_loss(y, hdw=hdw)

    if scenario in LONG_TERM_SCENARIOS:
        return _long_term_loss(y, pids, hdw=hdw)
    if scenario in SHORT_TERM_SCENARIOS:
        return _short_term_loss(y, cids, invalid_same_pid_mask, hdw=hdw)
    raise ValueError(f"Unsupported SAID scenario: {scenario}")
