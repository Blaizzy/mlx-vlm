"""Learning-rate schedules with optional linear warmup."""

import math

import mlx.optimizers as opt

SCHEDULE_NAMES = ("cosine_decay", "exponential_decay", "linear_schedule", "step_decay")


def total_optimizer_updates(iters: int, accumulation_steps: int = 1) -> int:
    """Optimizer updates for ``iters`` microsteps, including a final partial one."""
    return -(-iters // accumulation_steps)


def build_schedule(schedule_config: dict):
    """Build ``schedule_config["name"]`` from ``opt.schedulers`` with warmup.

    ``arguments[0]`` is the initial (peak) learning rate. Optional ``warmup``
    steps ramp linearly from ``warmup_init`` to it, then the schedule takes over.
    """
    name = schedule_config.get("name")
    if name not in SCHEDULE_NAMES:
        raise ValueError(f"lr schedule name must be one of {', '.join(SCHEDULE_NAMES)}.")
    arguments = schedule_config.get("arguments")
    if not isinstance(arguments, (list, tuple)) or not arguments:
        raise ValueError("lr schedule requires a non-empty 'arguments' list.")
    initial_lr = arguments[0]
    if not isinstance(initial_lr, (int, float)) or not math.isfinite(initial_lr) or initial_lr <= 0:
        raise ValueError("lr schedule initial learning rate must be positive.")
    warmup_steps = schedule_config.get("warmup", 0)
    warmup_init = schedule_config.get("warmup_init", 0.0)
    if not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError("lr schedule warmup must be a nonnegative integer.")
    if not math.isfinite(warmup_init) or warmup_init < 0:
        raise ValueError("lr schedule warmup_init must be nonnegative and finite.")
    bound = getattr(opt.schedulers, name)(*arguments)
    if warmup_steps:
        warmup_fn = opt.schedulers.linear_schedule(warmup_init, initial_lr, warmup_steps)
        return opt.schedulers.join_schedules([warmup_fn, bound], [warmup_steps + 1])
    return bound


def resolve_schedule_config(
    schedule,
    *,
    learning_rate: float,
    total_updates: int,
    warmup_steps: int = 0,
    warmup_init: float = 0.0,
    min_learning_rate: float = 0.0,
) -> dict | None:
    """Normalize a name or dict into a full config, filling defaults.

    A bare ``cosine_decay`` decays from ``learning_rate`` to
    ``min_learning_rate`` over the updates left after warmup. Returns ``None``
    when no schedule and no warmup is requested (constant rate).
    """
    if isinstance(schedule, str):
        schedule = {"name": schedule}
    if schedule is None:
        if not warmup_steps:
            return None
        schedule = {"name": "linear_schedule"}
        schedule["arguments"] = [learning_rate, learning_rate, 1]
    if not isinstance(schedule, dict):
        raise ValueError("lr_schedule must be a schedule name or a mapping.")
    config = dict(schedule)
    if warmup_steps and "warmup" not in config:
        config["warmup"] = warmup_steps
    if "warmup_init" not in config and warmup_steps:
        config["warmup_init"] = warmup_init
    if "arguments" not in config:
        if config.get("name") != "cosine_decay":
            raise ValueError("lr schedule 'arguments' are required for this schedule.")
        decay = max(total_updates - config.get("warmup", 0), 1)
        config["arguments"] = [learning_rate, decay, min_learning_rate]
    if config.get("warmup", 0) > total_updates:
        raise ValueError(
            f"lr warmup ({config['warmup']}) exceeds optimizer updates ({total_updates})."
        )
    return config
