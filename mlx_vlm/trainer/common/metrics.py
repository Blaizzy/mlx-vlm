"""Loss-derived metrics and formatting for shared training reports."""

import math
from collections.abc import Mapping

import mlx.core as mx


def cross_entropy_metrics(loss: float) -> dict[str, float]:
    """Return perplexity for a mean token negative log-likelihood in nats."""
    try:
        perplexity = math.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return {"perplexity": perplexity}


def normalize_loss_output(output):
    """Normalize scalar metrics, accepting previous (loss, weight) callables."""
    loss, metrics = output if isinstance(output, tuple) else (output, {})
    loss = loss if isinstance(loss, mx.array) else mx.array(loss)
    if loss.size != 1:
        raise ValueError("Training loss must be a scalar mean.")
    if not isinstance(metrics, Mapping):
        metrics = {"weight": metrics}
    normalized = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or key == "loss":
            raise ValueError("Loss metric names must be strings other than 'loss'.")
        value = value if isinstance(value, mx.array) else mx.array(value)
        if value.size != 1:
            raise ValueError(f"Loss metric {key!r} must be a scalar.")
        normalized[key] = mx.stop_gradient(value.reshape(()))
    normalized.setdefault("weight", mx.array(1, dtype=mx.int32))
    return loss.reshape(()), normalized


class MetricAccumulator:
    """Accumulate scalar auxiliary metrics without synchronizing each batch.

    Names starting with num_ are additive counts; other metrics are weighted
    means with their own denominators, so optional metrics can be omitted.
    """

    def __init__(self):
        self.sums, self.weights, self.counts = {}, {}, {}
        self.weight = mx.array(0.0)

    def add(self, loss, metrics):
        loss, metrics = normalize_loss_output((loss, metrics))
        weight = metrics["weight"].astype(mx.float32)
        self.weight += weight
        for key, value in {"loss": loss, **metrics}.items():
            if key == "weight":
                continue
            if key.startswith("num_"):
                self.counts[key] = self.counts.get(key, mx.array(0)) + value
            else:
                value = value.astype(mx.float32)
                self.sums[key] = self.sums.get(key, mx.array(0.0)) + value * weight
                self.weights[key] = self.weights.get(key, mx.array(0.0)) + weight

    def state(self):
        return self.sums, self.weights, self.counts, self.weight

    def compute(self):
        """Reduce across workers on CPU and materialize at reporting intervals."""
        with mx.stream(mx.cpu):
            names = sorted(self.sums)
            packed = mx.stack(
                [self.weight]
                + [v for key in names for v in (self.sums[key], self.weights[key])]
            )
            reduced = mx.distributed.all_sum(packed, stream=mx.cpu).tolist()
            result = {"weight": reduced[0]}
            for index, key in enumerate(names):
                numerator, denominator = reduced[1 + 2 * index : 3 + 2 * index]
                result[key] = numerator / denominator if denominator else 0.0
            if self.counts:
                names = sorted(self.counts)
                counts = mx.distributed.all_sum(
                    mx.stack([self.counts[key] for key in names]), stream=mx.cpu
                ).tolist()
                result.update(zip(names, counts))
        return result


def report_loss_metrics(values, prefix):
    """Prefix arbitrary loss metrics and derive perplexity from averaged NLL."""
    result = {
        f"{prefix}_{key}": value for key, value in values.items() if key != "weight"
    }
    if "nll" in values:
        result.update(
            {
                f"{prefix}_{key}": value
                for key, value in cross_entropy_metrics(values["nll"]).items()
            }
        )
    return result


def format_metric_report(info: dict, label: str) -> str:
    """Print every metric dynamically, four name/value pairs per line."""
    fields = []
    for name, value in info.items():
        if name == "iteration":
            continue
        formatted = f"{value:,}" if isinstance(value, int) else f"{value:.5g}"
        fields.append(f"{name}={formatted}")
    lines = [", ".join(fields[start : start + 4]) for start in range(0, len(fields), 4)]
    return f"{label} (iteration {info.get('iteration', 0)}):\n" + "\n".join(lines)
