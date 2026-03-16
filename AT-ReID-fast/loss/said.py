import torch
import torch.nn.functional as F


HEAD_SPECS = (
    {"name": "dt-st", "task": "dt", "time": "st", "modality": "rgb", "target": "cid"},
    {"name": "dt-lt", "task": "dt", "time": "lt", "modality": "rgb", "target": "pid"},
    {"name": "nt-st", "task": "nt", "time": "st", "modality": "ir", "target": "cid"},
    {"name": "nt-lt", "task": "nt", "time": "lt", "modality": "ir", "target": "pid"},
    {"name": "ad-st", "task": "ad", "time": "st", "modality": "all", "target": "cid"},
    {"name": "ad-lt", "task": "ad", "time": "lt", "modality": "all", "target": "pid"},
)
HEAD_SCENARIOS = tuple(spec["name"] for spec in HEAD_SPECS)
SCENARIO_TO_SPEC = {spec["name"]: spec for spec in HEAD_SPECS}
SHORT_TERM_SCENARIOS = {spec["name"] for spec in HEAD_SPECS if spec["target"] == "cid"}
LONG_TERM_SCENARIOS = {spec["name"] for spec in HEAD_SPECS if spec["target"] == "pid"}


def _gather_nll(log_probs, targets):
    return -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)


def _select_scenario_samples(y, pids, cids, mids, scenario: str):
    spec = SCENARIO_TO_SPEC[scenario]
    modality = spec["modality"]
    if modality == "rgb":
        mask = mids == 1
        return y[mask], pids[mask], cids[mask], mids[mask]
    if modality == "ir":
        mask = mids == 2
        return y[mask], pids[mask], cids[mask], mids[mask]
    return y, pids, cids, mids


def _empty_loss(y, *, hdw: bool):
    return y.new_zeros((0,)) if hdw else y.new_zeros(())


def _long_term_loss(logits, pids, *, hdw: bool):
    if logits.numel() == 0:
        return _empty_loss(logits, hdw=hdw)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    losses = _gather_nll(log_probs, pids)
    return losses if hdw else losses.mean()


def _short_term_loss(logits, cids, invalid_mask, *, hdw: bool):
    if logits.numel() == 0:
        return _empty_loss(logits, hdw=hdw)
    masked_logits = logits.float().masked_fill(invalid_mask, -1e4)
    log_probs = F.log_softmax(masked_logits, dim=-1)
    losses = _gather_nll(log_probs, cids)
    return losses if hdw else losses.mean()


def compute_head_losses(head_logits, pids, cids, mids, invalid_same_pid_mask, *, said: bool, hdw: bool):
    if not said:
        return tuple(_long_term_loss(logits, pids, hdw=hdw) for logits in head_logits)

    rgb_mask = mids == 1
    ir_mask = mids == 2
    all_invalid_mask = invalid_same_pid_mask.index_select(0, cids)
    modality_targets = {
        "all": (pids, cids, all_invalid_mask),
        "rgb": (pids[rgb_mask], cids[rgb_mask], all_invalid_mask[rgb_mask]),
        "ir": (pids[ir_mask], cids[ir_mask], all_invalid_mask[ir_mask]),
    }

    losses = []
    for logits, spec in zip(head_logits, HEAD_SPECS):
        modality = spec["modality"]
        target_type = spec["target"]
        target_pids, target_cids, invalid_mask = modality_targets[modality]
        if modality == "rgb":
            logits = logits[rgb_mask]
        elif modality == "ir":
            logits = logits[ir_mask]
        if target_type == "pid":
            losses.append(_long_term_loss(logits, target_pids, hdw=hdw))
        else:
            losses.append(_short_term_loss(logits, target_cids, invalid_mask, hdw=hdw))
    return tuple(losses)


def saidloss(y, pids, cids, mids, invalid_same_pid_mask, scenario, said=False, hdw=False):
    if not said:
        scenario = 'ad-lt'

    y, pids, cids, mids = _select_scenario_samples(y, pids, cids, mids, scenario)
    if y.numel() == 0:
        return _empty_loss(y, hdw=hdw)

    if scenario in LONG_TERM_SCENARIOS:
        return _long_term_loss(y, pids, hdw=hdw)
    if scenario in SHORT_TERM_SCENARIOS:
        return _short_term_loss(y, cids, invalid_same_pid_mask.index_select(0, cids), hdw=hdw)
    raise ValueError(f"Unsupported SAID scenario: {scenario}")
