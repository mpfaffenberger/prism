"""The eval runner's command line and model plan: the same settings the bot reads, from the same environment."""

from __future__ import annotations

import pytest

from evals.run import arguments, plan_from_env
from tests.reference_profile import DEEPSEEK, QWEN
from tests.test_reasoning import effort


def test_the_time_limits_default_to_the_bots_env_included() -> None:
    """From the bot's review of #33: Settings' class defaults ignored PRISM_*_DEADLINE_MINUTES."""
    assert (arguments([], {}).soft_minutes, arguments([], {}).hard_minutes) == (15, 20)
    env = {"PRISM_SOFT_DEADLINE_MINUTES": "3", "PRISM_HARD_DEADLINE_MINUTES": "7"}
    args = arguments([], env)
    assert (args.soft_minutes, args.hard_minutes) == (3, 7)


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
