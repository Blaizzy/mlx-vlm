from mlx_vlm.server.prefill_scheduler import (
    ResumablePrefill,
    TailFairScheduler,
    TailWork,
)


def item(name, prompt, cached=0):
    return ResumablePrefill(
        name,
        TailWork(prompt, cached),
        cache_state=object(),
        remaining_input=object(),
    )


def test_scheduler_charges_suffix_not_prompt():
    assert TailWork(30_000, 29_988).uncached_tokens == 12


def test_scheduler_runs_shortest_tail_first():
    scheduler = TailFairScheduler(slice_tokens=64)
    long, hit = item("long", 4096), item("hit", 30_000, 29_988)
    scheduler.enqueue(long)
    scheduler.enqueue(hit)

    selected, budget = scheduler.choose()

    assert selected is hit
    assert budget == 12


def test_scheduler_commit_is_transactional():
    pending = item("pending", 100)
    pending.commit_segment(64, "suffix")

    assert pending.processed_tokens == 64
    assert pending.remaining_tokens == 36

    try:
        pending.commit_segment(37, "invalid")
    except ValueError:
        pass
    else:
        raise AssertionError("overflow segment accepted")

    assert pending.processed_tokens == 64


def test_scheduler_bounds_starvation():
    scheduler = TailFairScheduler(slice_tokens=32, max_deferrals=1)
    long, first, second = item("long", 1000), item("first", 10), item("second", 11)
    scheduler.enqueue(long)
    scheduler.enqueue(first)
    scheduler.enqueue(second)

    assert scheduler.choose()[0] is first
    scheduler.requeue(first)
    assert scheduler.choose()[0] is long


def test_scheduler_cancel_is_request_local():
    scheduler = TailFairScheduler(slice_tokens=32)
    first, second = item("first", 100), item("second", 100)
    scheduler.enqueue(first)
    scheduler.enqueue(second)

    assert scheduler.cancel("first") is first
    assert tuple(scheduler.pending_ids()) == ("second",)
    assert scheduler.cancel("missing") is None


def test_scheduler_drain_detaches_resident_work():
    scheduler = TailFairScheduler(slice_tokens=32)
    first, second = item("first", 100), item("second", 100)
    scheduler.enqueue(first)
    scheduler.enqueue(second)

    assert scheduler.drain() == (first, second)
    assert len(scheduler) == 0
    assert scheduler.drain() == ()
