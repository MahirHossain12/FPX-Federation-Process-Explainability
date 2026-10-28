
def get_consecutive_excluded_count(history, round_idx, client_id):
    count = 0
    for i in range(round_idx, -1, -1):
        if history[i]["excluded"][client_id]:
            count += 1
        else:
            break
    return count

def get_notable_clients(record, z_threshold=-1.0, warmup_rounds=5):
    """Discover which clients look anomalous THIS round, purely from the signals.
    Mirrors the aggregation logic's warmup period: early rounds are known to be
    volatile before the federation stabilizes, so we do not narrate single-round
    noise during warmup as if it were a real signal - this keeps the explanation
    consistent with what the system actually acts on."""
    if record["round"] <= warmup_rounds:
        return []
    notable = []
    for i, z in enumerate(record["z_score"]):
        if z < z_threshold or record["excluded"][i]:
            notable.append(i)
    return notable

def explain_contribution(record):
    shares = record["effective_share"]
    max_i, min_i = shares.index(max(shares)), shares.index(min(shares))
    if max(shares) - min(shares) < 0.05:
        return "All sites contributed roughly equally to the global model this round."
    return (f"Site {max_i} contributed the most to this round's model update "
            f"({shares[max_i]*100:.0f}% effective weight), while Site {min_i} "
            f"contributed the least ({shares[min_i]*100:.0f}%).")

def explain_divergence(record, client_id, excluded_this_round):
    z = record["z_score"][client_id]
    if excluded_this_round:
        return (f"Site {client_id}'s update looked substantially different from the rest of "
                f"the federation this round (divergence score {z:.2f}, well outside the normal range).")
    else:
        return (f"Site {client_id}'s update has just started looking unusual compared to the rest of "
                f"the federation this round (divergence score {z:.2f}). Not yet acted on - "
                f"the system waits for a consistent pattern before down-weighting a site.")

def explain_exclusion(history, round_idx, client_id):
    record = history[round_idx]
    if not record["excluded"][client_id]:
        return None
    streak = get_consecutive_excluded_count(history, round_idx, client_id)
    return (f"Site {client_id}'s contribution was significantly down-weighted this round. "
            f"This site's updates have looked unusual for {streak} consecutive round(s) - "
            f"this may indicate a data quality or labeling issue at this site, and is worth investigating "
            f"before trusting its future contributions.")

def explain_instability(history, round_idx, drop_threshold=0.015):
    if round_idx == 0:
        return None
    delta = history[round_idx]["test_accuracy"] - history[round_idx-1]["test_accuracy"]
    if delta < -drop_threshold:
        anomalous = [i for i, z in enumerate(history[round_idx]["z_score"]) if z < -1.0]
        cause = f", coinciding with unusual updates from Site(s) {anomalous}" if anomalous else " with no single site clearly responsible"
        return f"Global accuracy dropped by {abs(delta)*100:.1f} points this round{cause}."
    return None

def generate_round_briefing(history, round_idx, z_threshold=-1.0, warmup_rounds=5):
    record = history[round_idx]

    lines = [f"--- Round {record['round']} briefing ---"]
    lines.append(f"Global test accuracy: {record['test_accuracy']*100:.1f}%")
    lines.append(explain_contribution(record))

    if record["round"] <= warmup_rounds:
        lines.append("Federation still stabilizing (warmup period) - anomaly detection not yet active.")
    else:
        notable = get_notable_clients(record, z_threshold=z_threshold, warmup_rounds=warmup_rounds)
        if not notable:
            lines.append("No sites showed unusual update behavior this round.")
        else:
            for cid in notable:
                lines.append(explain_divergence(record, cid, record["excluded"][cid]))
                excl_text = explain_exclusion(history, round_idx, cid)
                if excl_text:
                    lines.append(excl_text)

    instab_text = explain_instability(history, round_idx)
    if instab_text:
        lines.append(instab_text)

    return "\n".join(lines)
