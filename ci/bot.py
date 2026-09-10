from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


class BotOutputError(ValueError):
    pass


class ComponentOutput(Protocol):
    component_names: frozenset[str]

    def sections(self, record: Mapping[str, Any]) -> Sequence["BotSection"]: ...


@dataclass(frozen=True)
class BotStage:
    name: str
    status: str
    detail: str = ""


@dataclass(frozen=True)
class BotMetric:
    name: str
    base: Any
    head: Any
    change_pct: Any
    verdict: str
    unit: str = ""


@dataclass(frozen=True)
class BotSection:
    title: str
    component: str
    status: str
    paths: tuple[str, ...] = ()
    stages: tuple[BotStage, ...] = ()
    metrics: tuple[BotMetric, ...] = ()
    messages: tuple[str, ...] = ()

    def render(self) -> list[str]:
        tag = "<details>" if self.status in {"Passed", "Improved"} else "<details open>"
        lines = [
            tag,
            (
                f"<summary>{_icon(self.status)} <strong>{_cell(self.title)}</strong> · "
                f"{_cell(self.component)} · {_cell(self.status)}</summary>"
            ),
            "",
        ]
        if self.paths:
            shown = ", ".join(f"`{_cell(path)}`" for path in self.paths[:8])
            if len(self.paths) > 8:
                shown += f", and {len(self.paths) - 8} more"
            lines.extend([f"Changed: {shown}", ""])
        if self.stages:
            lines.extend(["| Check | Result | Details |", "|---|---|---|"])
            lines.extend(
                f"| {_cell(stage.name)} | {_cell(stage.status)} | {_cell(stage.detail)} |"
                for stage in self.stages
            )
        if self.metrics:
            lines.extend(
                [
                    "",
                    "| Metric | Main | PR | Change | Verdict |",
                    "|---|---:|---:|---:|---|",
                ]
            )
            lines.extend(
                "| "
                + " | ".join(
                    (
                        _cell(_metric_name(metric.name)),
                        _cell(_measurement(metric.base, metric.unit)),
                        _cell(_measurement(metric.head, metric.unit)),
                        _cell(_change(metric.change_pct)),
                        _cell(metric.verdict),
                    )
                )
                + " |"
                for metric in self.metrics
            )
        if self.messages:
            lines.append("")
            lines.extend(f"- {_cell(message)}" for message in self.messages)
        lines.extend(["", "</details>"])
        return lines


class ModelPathOutput:
    """Render one independent section for every affected model family."""

    component_names = frozenset({"model_path", "new_model_path"})
    phase_order = ("synthetic", "hf_checkpoint")

    def sections(self, record: Mapping[str, Any]) -> Sequence[BotSection]:
        return tuple(
            self._section(record, model) for model in sorted(self._models(record))
        )

    def _models(self, record: Mapping[str, Any]) -> set[str]:
        models = {
            str(item["model"])
            for key in ("jobs", "gates", "results")
            for item in _items(record, key)
            if item.get("component") in self.component_names and item.get("model")
        }
        models.update(
            str(error["subject"])
            for error in _items(record, "errors")
            if error.get("component") in self.component_names and error.get("subject")
        )
        return models

    def _section(self, record: Mapping[str, Any], model: str) -> BotSection:
        jobs = self._jobs(record, model)
        gates = self._matching(record, "gates", model)
        errors = self._errors(record, model)
        results = self._matching(record, "results", model)
        blocked = (
            record.get("kind") == "ci_execution" and record.get("outcome") == "blocked"
        )
        return BotSection(
            title=model,
            component="ModelPath",
            status=self._status(record, jobs, gates, errors, results),
            paths=_paths((*jobs, *gates, *errors)),
            stages=self._stages(jobs, gates, errors, results, blocked),
            metrics=self._metrics(results),
            messages=self._messages(jobs, gates, errors, results, blocked),
        )

    def _matching(
        self, record: Mapping[str, Any], key: str, model: str
    ) -> list[Mapping[str, Any]]:
        return [
            item
            for item in _items(record, key)
            if item.get("component") in self.component_names
            and str(item.get("model")) == model
        ]

    def _errors(self, record: Mapping[str, Any], model: str) -> list[Mapping[str, Any]]:
        return [
            error
            for error in _items(record, "errors")
            if error.get("component") in self.component_names
            and str(error.get("subject")) == model
        ]

    def _jobs(self, record: Mapping[str, Any], model: str) -> list[Mapping[str, Any]]:
        jobs = self._matching(record, "jobs", model)
        for gate in self._matching(record, "gates", model):
            pending = gate.get("pending_work")
            if isinstance(pending, Mapping):
                jobs.append(pending)
        return list({str(job.get("id", model)): job for job in jobs}.values())

    @staticmethod
    def _status(
        record: Mapping[str, Any],
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
    ) -> str:
        if errors:
            return "Blocked"
        outcome = _first_outcome(results)
        if outcome:
            return _label(outcome)
        if any(gate.get("status") == "awaiting_maintainer_approval" for gate in gates):
            return "Awaiting maintainer approval"
        if record.get("kind") == "approved_job_plan" and jobs:
            return "Ready for runner dispatch"
        if (
            record.get("kind") == "ci_execution"
            and record.get("outcome") == "blocked"
            and jobs
        ):
            return "Not run"
        return "Awaiting /ci run" if jobs else "No jobs"

    def _stages(
        self,
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        blocked: bool,
    ) -> tuple[BotStage, ...]:
        jobs_by_phase = {phase: job for job in jobs for phase in _job_phases(job)}
        results_by_phase = {
            str(result["phase"]): result
            for result in _phase_results(results)
            if result.get("phase")
        }
        error_phases = {
            str(details["mode"])
            for error in errors
            if isinstance((details := error.get("details")), Mapping)
            and details.get("mode")
        }
        phases = set(jobs_by_phase) | set(results_by_phase) | error_phases
        ordered = [phase for phase in self.phase_order if phase in phases]
        ordered.extend(sorted(phases - set(ordered)))
        awaiting = any(
            gate.get("status") == "awaiting_maintainer_approval" for gate in gates
        )
        stages = []
        for phase in ordered:
            result = results_by_phase.get(phase)
            if phase in error_phases:
                status = "Blocked"
            elif result:
                status = _label(str(result.get("outcome", "unknown")))
            elif awaiting:
                status = "Awaiting approval"
            elif blocked:
                status = "Not run"
            else:
                status = "Planned"
            stages.append(
                BotStage(
                    _phase_name(phase),
                    status,
                    _phase_detail(jobs_by_phase.get(phase), phase),
                )
            )
        return tuple(stages)

    @staticmethod
    def _metrics(
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[BotMetric, ...]:
        metrics = []
        for result in _phase_results(results):
            findings = result.get("findings")
            values = result.get("metrics")
            if not isinstance(values, Mapping) and isinstance(findings, Mapping):
                values = findings.get("metrics")
            if not isinstance(values, Mapping):
                continue
            correctness = (
                findings.get("correctness") if isinstance(findings, Mapping) else None
            )
            advisory = _correctness_failed(correctness)
            for name, value in sorted(values.items()):
                if not isinstance(value, Mapping):
                    continue
                metrics.append(
                    BotMetric(
                        str(name),
                        value.get("base"),
                        value.get("head"),
                        value.get("change_pct"),
                        "advisory" if advisory else str(value.get("verdict", "")),
                        str(value.get("unit", "")),
                    )
                )
        return tuple(metrics)

    def _messages(
        self,
        jobs: Sequence[Mapping[str, Any]],
        gates: Sequence[Mapping[str, Any]],
        errors: Sequence[Mapping[str, Any]],
        results: Sequence[Mapping[str, Any]],
        blocked: bool,
    ) -> tuple[str, ...]:
        messages = [str(error.get("code", "unknown_error")) for error in errors]
        messages.extend(
            f"{_phase_name(str(phase))} unavailable: {reason}."
            for job in jobs
            for phase, reason in (
                job.get("unavailable_phases", {}).items()
                if isinstance(job.get("unavailable_phases"), Mapping)
                else ()
            )
        )
        for result in results:
            messages.extend(_execution_messages(result))
            failure = result.get("checkpoint_failure")
            if isinstance(failure, Mapping) and failure.get("code"):
                messages.append(_failure_message(str(failure["code"])))
            if result.get("outcome") == "no_eligible_runner":
                messages.extend(_no_runner_messages(result))
        for result in _phase_results(results):
            message = _findings_message(result)
            if message:
                messages.append(message)
        if any(gate.get("status") == "awaiting_maintainer_approval" for gate in gates):
            messages.append(
                "Configuration is valid. No Apple Silicon job starts until a "
                "maintainer approves this new model run."
            )
        if not gates and jobs and not errors and not results:
            messages.append("The immutable job manifest is ready for runner dispatch.")
        if blocked and jobs and not errors and not results:
            messages.append("Not run because another check blocked this attempt.")
        return tuple(messages)


class DocsChangeOutput:
    """Render documentation checks performed on GitHub-hosted runners."""

    component_names = frozenset({"docs_change"})

    def sections(self, record: Mapping[str, Any]) -> Sequence[BotSection]:
        checks = [
            item
            for item in _items(record, "checks")
            if item.get("component") == "docs_change"
        ]
        results = [
            item
            for item in _items(record, "results")
            if item.get("component") == "docs_change"
        ]
        errors = [
            item
            for item in _items(record, "errors")
            if item.get("component") == "docs_change"
        ]
        if not checks and not results and not errors:
            return ()
        status = "Blocked" if errors else _label(_first_outcome(results) or "queued")
        findings = results[-1].get("findings") if results else None
        messages = (
            tuple(str(value) for value in findings.get("new_errors", [])[:8])
            if isinstance(findings, Mapping)
            else ()
        )
        return (
            BotSection(
                "Documentation",
                "DocsChange",
                status,
                paths=_paths((*checks, *results, *errors)),
                stages=(
                    BotStage(
                        "Documentation",
                        status,
                        "local links and MkDocs navigation",
                    ),
                ),
                messages=messages,
            ),
        )


class BotOutput:
    """Compose a concise comment from independent CI component sections."""

    def __init__(
        self,
        record: Mapping[str, Any],
        components: Sequence[ComponentOutput] | None = None,
    ):
        self.record = record
        if components is None:
            from ci.components.registry import outputs

            components = outputs()
        self.components = tuple(components)

    def render(self) -> str:
        sections = tuple(
            section
            for component in self.components
            for section in component.sections(self.record)
        )
        self._reject_unknown_components()
        status = self._status(sections)
        summary = (
            f"{_icon(status)} **{_cell(status)}** · "
            f"Commit: `{_cell(self.record['head_sha'])}`"
        )
        attempt = self.record.get("attempt_id")
        if self.record.get("kind") == "ci_execution" and attempt:
            summary += f" · Attempt: `{_cell(attempt)}`"
        if run_url := self.record.get("run_url"):
            summary += f" · [View run]({_url(run_url)})"
        lines = [self._marker(), summary]
        for section in sections:
            lines.extend(["", *section.render()])
        if not sections:
            lines.extend(self._empty_output())
        return "\n".join(lines) + "\n"

    def _marker(self) -> str:
        attempt = self.record.get("attempt_id")
        if self.record.get("kind") == "ci_execution" and attempt:
            return f"<!-- mlx-vlm:ci:attempt:{_cell(attempt)} -->"
        return "<!-- mlx-vlm:ci:plan -->"

    def _status(self, sections: Sequence[BotSection]) -> str:
        if self.record.get("outcome") == "blocked":
            return "Blocked"
        statuses = {section.status for section in sections}
        for status in (
            "Blocked",
            "Test failed",
            "Regressed",
            "No eligible runner",
            "Infrastructure failed",
            "Cancelled",
            "Running",
            "Queued",
            "Coalesced",
            "Awaiting maintainer approval",
            "Awaiting /ci run",
            "Ready for runner dispatch",
            "Improved",
            "Passed",
        ):
            if status in statuses:
                return status
        return _label(str(self.record.get("outcome", "unknown")))

    def _empty_output(self) -> list[str]:
        errors = _items(self.record, "errors")
        if not errors:
            return ["", "No Apple Silicon jobs are required for this change."]
        lines = ["", "| Component | Subject | Problem |", "|---|---|---|"]
        lines.extend(
            f"| {_cell(error.get('component', 'planner'))} | "
            f"{_cell(error.get('subject', 'pull_request'))} | "
            f"{_cell(error.get('code', 'unknown_error'))} |"
            for error in errors
        )
        return lines

    def _reject_unknown_components(self) -> None:
        supported = set().union(
            *(component.component_names for component in self.components)
        )
        encountered = {
            str(item["component"])
            for key in ("jobs", "gates", "checks", "results")
            for item in _items(self.record, key)
            if item.get("component")
        }
        encountered.update(
            str(error["component"])
            for error in _items(self.record, "errors")
            if error.get("component") not in {None, "planner"}
        )
        components = self.record.get("components")
        if isinstance(components, Sequence) and not isinstance(
            components, (str, bytes)
        ):
            encountered.update(str(value) for value in components if value)
        if unknown := sorted(encountered - supported):
            raise BotOutputError(
                "no bot output renderer for components: " + ", ".join(unknown)
            )


def _items(record: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = record.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _job_phases(job: Mapping[str, Any]) -> tuple[str, ...]:
    phases = job.get("phases")
    if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)):
        return tuple(str(phase) for phase in phases)
    return (str(job["mode"]),) if job.get("mode") else ()


def _phase_results(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    expanded = []
    for result in results:
        phases = result.get("phases")
        if isinstance(phases, Mapping):
            for name, value in phases.items():
                if isinstance(value, Mapping):
                    phase = dict(value)
                    phase["phase"] = str(name)
                    if name == "hf_checkpoint":
                        phase.setdefault("cache", result.get("cache", {}))
                    expanded.append(phase)
        elif result.get("mode"):
            phase = dict(result)
            phase["phase"] = str(result["mode"])
            expanded.append(phase)
    return tuple(expanded)


def _first_outcome(results: Sequence[Mapping[str, Any]]) -> str | None:
    outcomes = {str(result.get("outcome", "")) for result in results}
    for outcome in (
        "test_failure",
        "regressed",
        "no_eligible_runner",
        "infrastructure_failure",
        "cancelled",
        "running",
        "queued",
        "coalesced",
        "improved",
        "passed",
    ):
        if outcome in outcomes:
            return outcome
    return None


def _paths(items: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    paths = set()
    for item in items:
        paths.update(str(path) for path in item.get("changed_paths", []))
        details = item.get("details")
        if isinstance(details, Mapping):
            paths.update(str(path) for path in details.get("changed_paths", []))
    return tuple(sorted(paths))


def _phase_detail(job: Mapping[str, Any] | None, phase: str) -> str:
    if not job:
        return ""
    value = job.get(phase)
    if not isinstance(value, Mapping):
        return str(job.get("id", ""))
    if phase == "synthetic":
        return " / ".join(
            str(value[key]) for key in ("adapter", "profile") if value.get(key)
        )
    repository = str(value.get("repo", ""))
    revision = str(value.get("revision", ""))
    return f"{repository}@{revision[:12]}" if revision else repository


def _execution_messages(result: Mapping[str, Any]) -> tuple[str, ...]:
    selected = result.get("selected_device")
    runner = (
        selected.get("name", "unknown")
        if isinstance(selected, Mapping)
        else result.get("device", "not allocated")
    )
    cache = result.get("cache")
    cache_state = "not checked"
    if isinstance(cache, Mapping) and cache:
        cache_state = (
            "reused"
            if cache.get("reused")
            else str(cache.get("after", cache.get("before", "not checked")))
        )
    correctness = []
    has_metrics = False
    for phase in _phase_results((result,)):
        findings = phase.get("findings")
        if not isinstance(findings, Mapping):
            continue
        value = findings.get("correctness")
        if isinstance(value, Mapping) and isinstance(value.get("match"), bool):
            correctness.append(not _correctness_failed(value))
        has_metrics = has_metrics or isinstance(findings.get("metrics"), Mapping)
    correctness_state = (
        "failed"
        if False in correctness
        else "passed" if correctness else "not reported"
    )
    performance = (
        "advisory"
        if False in correctness and has_metrics
        else "reported" if has_metrics else "not run"
    )
    return (
        f"Runner: {runner} · Cache: {cache_state}.",
        f"Correctness: {correctness_state} · Performance: {performance} · "
        f"Terminal state: {_label(str(result.get('outcome', 'unknown')))}.",
    )


def _findings_message(result: Mapping[str, Any]) -> str | None:
    findings = result.get("findings")
    if not isinstance(findings, Mapping):
        return None
    phase = str(result.get("phase", "hf_checkpoint"))
    if error := findings.get("error"):
        return f"{_phase_name(phase)} failed: {error}."
    correctness = findings.get("correctness")
    if _correctness_failed(correctness):
        if phase == "synthetic":
            return "Synthetic outputs or parameter structure differ from main."
        return "Checkpoint correctness failed; performance numbers are advisory."
    if isinstance(correctness, Mapping) and correctness.get("match") is True:
        return f"{_phase_name(phase)} correctness passed."
    return None


def _correctness_failed(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        value.get("match") is False or value.get("contracts_pass") is False
    )


def _failure_message(reason: str) -> str:
    return {
        "checkpoint_not_found": "The configured checkpoint or revision was not found.",
        "access_denied": "The configured checkpoint requires access the CI runner does not have.",
        "disk_full": "The selected runner does not have enough disk space.",
        "network_transient": "The checkpoint download failed temporarily; retry with /ci run.",
        "checkpoint_policy_failed": "The runner rejected the checkpoint because it failed integrity policy.",
        "checkpoint_internal_error": "The runner could not safely prepare the checkpoint.",
    }.get(reason, "The model run failed.")


def _no_runner_messages(result: Mapping[str, Any]) -> tuple[str, ...]:
    memory = result.get("required_memory_gib", "unknown")
    disk = result.get("required_disk_gib")
    requirement = f"{memory} GiB memory"
    if disk is not None:
        requirement += f" and {disk} GiB disk"
    return (
        f"No eligible Apple Silicon runner is available. Required: {requirement}.",
        "Retry with /ci run.",
    )


def _phase_name(value: str) -> str:
    return {
        "synthetic": "Synthetic",
        "hf_checkpoint": "HF checkpoint",
    }.get(value, value.replace("_", " ").title())


def _label(value: str) -> str:
    return {
        "blocked": "Blocked",
        "ready": "Ready",
        "queued": "Queued",
        "coalesced": "Coalesced",
        "running": "Running",
        "passed": "Passed",
        "improved": "Improved",
        "regressed": "Regressed",
        "no_eligible_runner": "No eligible runner",
        "test_failure": "Test failed",
        "infrastructure_failure": "Infrastructure failed",
        "cancelled": "Cancelled",
    }.get(value, value.replace("_", " ").title())


def _icon(status: str) -> str:
    return {
        "Passed": "✅",
        "Improved": "✅",
        "Test failed": "❌",
        "Regressed": "❌",
        "Blocked": "⛔",
        "Infrastructure failed": "⚠️",
        "No eligible runner": "⚠️",
        "Cancelled": "⏹️",
        "Running": "🔄",
        "Queued": "⏳",
        "Coalesced": "↪️",
        "Awaiting maintainer approval": "⏳",
        "Awaiting /ci run": "⏳",
        "Ready for runner dispatch": "⏳",
        "Not run": "➖",
        "No jobs": "➖",
    }.get(status, "•")


def _measurement(value: Any, unit: str) -> str:
    if value is None:
        return "—"
    rendered = (
        f"{value:.3f}".rstrip("0").rstrip(".")
        if isinstance(value, float)
        else str(value)
    )
    return f"{rendered} {unit}".rstrip()


def _metric_name(value: str) -> str:
    return {
        "decode_tps": "Decode throughput",
        "embedding_latency_ms": "Embedding latency",
        "embedding_tps": "Embedding throughput",
        "peak_memory_gib": "Peak memory",
        "prefill_tps": "Prefill throughput",
        "ttft_ms": "TTFT",
        "wall_ms": "Wall time",
    }.get(value, value.replace("_", " ").title())


def _change(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:+.2f}%" if isinstance(value, (int, float)) else str(value)


def _cell(value: Any) -> str:
    text = str(value).replace("@", "@\u200b")
    return (
        text.replace("|", "\\|")
        .replace("`", "\\`")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\r", " ")
        .replace("\n", " ")[:200]
    )


def _url(value: Any) -> str:
    text = str(value)
    return text if text.startswith(("https://", "http://")) else "#"
