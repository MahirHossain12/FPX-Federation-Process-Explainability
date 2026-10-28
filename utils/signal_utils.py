
import torch
import numpy as np

def flat_params(state_dict):
    return torch.cat([v.float().view(-1) for v in state_dict.values()])

def compute_divergence_signals(client_states, global_state_before):
    """Per-client cosine similarity of their update vs the mean update this round,
    plus a z-score of how anomalous each client is relative to its peers THIS round."""
    global_flat = flat_params(global_state_before)
    deltas = [flat_params(s) - global_flat for s in client_states]
    mean_delta = torch.stack(deltas).mean(dim=0)
    sims = [torch.nn.functional.cosine_similarity(mean_delta.unsqueeze(0), d.unsqueeze(0)).item() for d in deltas]
    norms = [d.norm().item() for d in deltas]
    sims_arr = np.array(sims)
    mu, sigma = sims_arr.mean(), sims_arr.std() + 1e-8
    z_scores = ((sims_arr - mu) / sigma).tolist()
    return {"cos_sim": sims, "update_norm": norms, "z_score": z_scores}

def update_exclusion_state(z_scores, consecutive_low, round_num, warmup_rounds=5,
                            persistence_rounds=2, z_threshold=-1.0):
    """Stateful exclusion flagging: requires N consecutive anomalous rounds AFTER
    a warmup period, to avoid flagging normal early-training volatility as a fault."""
    excluded = [False] * len(z_scores)
    if round_num > warmup_rounds:
        for i, z in enumerate(z_scores):
            if z < z_threshold:
                consecutive_low[i] += 1
            else:
                consecutive_low[i] = 0
            if consecutive_low[i] >= persistence_rounds:
                excluded[i] = True
    else:
        for i in range(len(z_scores)):
            consecutive_low[i] = 0
    return excluded, consecutive_low

def compute_contribution_weights(base_weights, excluded, down_weight_factor=0.05):
    """Apply down-weighting to excluded clients' aggregation weight."""
    adj = [w * down_weight_factor if e else w for w, e in zip(base_weights, excluded)]
    total = sum(adj)
    effective_share = [a / total for a in adj]
    return adj, effective_share
