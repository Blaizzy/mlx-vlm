"""Lifetime counters used by CLI and server request reporting."""


def speculative_stats_snapshot(draft_model):
    return getattr(draft_model, "speculative_stats", (0, 0, 0))


def record_round(draft_model, proposals, emitted):
    rounds, accepted, drafted = speculative_stats_snapshot(draft_model)
    drafts = proposals.tolist()
    for proposal, output in zip(drafts, emitted):
        if not output:
            continue
        drafted += len(proposal)
        for expected, actual in zip(proposal, output):
            if expected != actual:
                break
            accepted += 1
    draft_model.speculative_stats = (rounds + 1, accepted, drafted)


def speculative_stats_since(draft_model, snapshot):
    values = tuple(
        a - b for a, b in zip(speculative_stats_snapshot(draft_model), snapshot)
    )
    return values if values[0] else (None, None, None)


def format_speculative_stats(draft_model):
    rounds, accepted, drafted = speculative_stats_snapshot(draft_model)
    if not rounds:
        return None
    rate = 100 * accepted / drafted if drafted else 0
    return f"MTP: {accepted}/{drafted} drafts accepted ({rate:.1f}%) in {rounds} rounds"
