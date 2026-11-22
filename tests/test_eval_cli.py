"""The eval runner's command line and model plan: the same settings the bot reads, from the same environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoreviewer.review.deadline import Deadline, checking_cutoff
from autoreviewer.review.job import CheckResult, Reviewed
from autoreviewer.review.pipeline import Outcome
from autoreviewer.review.schemas import Finding
from evals import run as runner
from evals.cases import Case
from evals.run import arguments, plan_from_env
from tests.reference_profile import DEEPSEEK, QWEN
from tests.review_fakes import finding
from tests.test_reasoning import effort


def test_the_time_limits_default_to_the_bots_env_included() -> None:
    """From the bot's review of #33: Settings' class defaults ignored PRISM_*_DEADLINE_MINUTES."""
    assert (arguments([], {}).soft_minutes, arguments([], {}).hard_minutes) == (15, 20)
    env = {"PRISM_SOFT_DEADLINE_MINUTES": "3", "PRISM_HARD_DEADLINE_MINUTES": "7"}
    args = arguments([], env)
    assert (args.soft_minutes, args.hard_minutes) == (3, 7)


@pytest.mark.anyio
async def test_a_case_runs_under_the_bots_checking_cutoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without it a case waits for every check, and one that reaches the hard stop scores as nothing found."""
    seen: list[Deadline] = []

    async def review_change(*, deadline: Deadline, **_: object) -> CheckResult:
        seen.append(deadline)
        return CheckResult("Nothing to review", "", "success")

    repo = runner.GitRepo(tmp_path, "b" * 40, "a" * 40)
    monkeypatch.setattr(runner, "build_repo", lambda case, parent: (repo, [1]))
    monkeypatch.setattr(runner, "review_change", review_change)
    case = Case("demo", "stats.py", "old", "new", "A change")
    await runner.run_case(case, lambda workspace: NotImplemented, [], tmp_path, soft_s=300, hard_s=400)
    [deadline] = seen
    assert (deadline.soft_s, deadline.checking_s) == (300, checking_cutoff(400))


@pytest.mark.anyio
async def test_a_case_cut_short_still_scores_what_it_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """From the bot's review of #65: a review the cutoff ended early has an outcome, and its hits count."""
    hit = Finding.model_validate({**finding(5), "kind": "correctness"})

    async def review_change(**_: object) -> Reviewed:
        return Reviewed(Outcome(inline=[hit], cut_short=True), None, None)

    repo = runner.GitRepo(tmp_path, "b" * 40, "a" * 40)
    monkeypatch.setattr(runner, "build_repo", lambda case, parent: (repo, [5]))
    monkeypatch.setattr(runner, "review_change", review_change)
    case = Case("demo", "stats.py", "old", "new", "A change")
    result = await runner.run_case(case, lambda workspace: NotImplemented, [], tmp_path, soft_s=300, hard_s=400)
    assert (result.posted_hit, result.blocking_hit, result.cut_short, result.error) == (True, True, True, None)


@pytest.mark.parametrize(
    ("argv", "error"),
    [
        (["--concurrency", "0"], "--concurrency must be at least 1"),
        (["--soft-minutes", "0"], "--soft-minutes must be greater than 0"),
        (["--soft-minutes", "50", "--hard-minutes", "50"], "below --hard-minutes"),
    ],
)
def test_bad_arguments_stop_the_run(argv: list[str], error: str, capsys: pytest.CaptureFixture[str]) -> None:
    """From the bot's review of #33: the checks were never run."""
    with pytest.raises(SystemExit):
        arguments(argv, {})
    assert error in capsys.readouterr().err


def test_the_plan_reads_the_bots_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """From the bot's reviews of #36 and #47: plan_from_env carries every setting the bot's own plan reads."""
    for name, value in {
        "BOODLETON_API_KEY": "k",
        "BOODLETON_BASE_URL": "http://boodleton.test/v1",
        "PRISM_MODELS": f"security={DEEPSEEK},security.verifier={QWEN}",
        "PRISM_MODEL_LANES": f"{QWEN}=2,{DEEPSEEK}=3",
        "PRISM_REASONING": "docs=high",
    }.items():
        monkeypatch.setenv(name, value)
    plan = plan_from_env()
    assert (plan.base_url, plan.api_key, dict(plan.lanes)) == ("http://boodleton.test/v1", "k", {QWEN: 2, DEEPSEEK: 3})
    assert (plan.models["security"], plan.models["security.verifier"]) == (DEEPSEEK, QWEN)
    assert (effort(plan.finder("docs")[0]), effort(plan.finder("tests")[0])) == ("high", None)
