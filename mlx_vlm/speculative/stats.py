"""Per-request speculative counters, owned by the request cache."""

from dataclasses import dataclass


@dataclass
class SpeculativeStats:
    rounds: int = 0
    accepted: int = 0
    drafted: int = 0

    def record(self, proposals, emitted):
        if not emitted or not proposals:
            return
        self.rounds += 1
        self.drafted += len(proposals)
        for expected, actual in zip(proposals, emitted):
            if expected != actual:
                break
            self.accepted += 1

    def snapshot(self):
        return self.rounds, self.accepted, self.drafted


def format_speculative_stats(stats):
    if stats is None or not stats[0]:
        return None
    rounds, accepted, drafted = stats
    rate = 100 * accepted / drafted if drafted else 0
    return f"MTP: {accepted}/{drafted} drafts accepted ({rate:.1f}%) in {rounds} rounds"
