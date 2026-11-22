"""A pool for finders with a dedicated checker: the pool places finders, every check goes to the checker unless the
finder is the checker's own model (an overflow finder, say), whose checks the pool places instead."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from autoreviewer.config_cli import main
from autoreviewer.model_config import PrismConfig
from autoreviewer.review.job import ReviewDeps
from autoreviewer.review.memory import ReviewMemory
from autoreviewer.review.workspace import Workspace
from autoreviewer.web.app import ModelPlan, models_note, overflow_deps
from autoreviewer.web.config import OVERFLOW_REPOS, Settings
from tests.test_prism_model_plan import APP_ENV

POOL = {"openai:a": 1, "openai:b": 2}
CHECKER = "openai:sol"


def settings_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **extra: object) -> Settings:
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    path = tmp_path / "prism.json"
    path.write_text(json.dumps({"pool": POOL, "checker": CHECKER, "checks": "checker", **extra}))
    return Settings.from_env({**APP_ENV, "PRISM_CONFIG": str(path), OVERFLOW_REPOS: "org/repo"})


def deps(tmp_path: Path, select: Mock | None = None) -> ReviewDeps:
    return ReviewDeps(
        Mock(), Mock(), Mock(), tmp_path, "bot", "normal", memory=ReviewMemory(tmp_path), select_correctness=select
    )


def test_pool_finders_are_checked_by_the_dedicated_checker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoreviewer.web.app import build_lenses, correctness_agents, simplifier_agents

    settings = settings_for(tmp_path, monkeypatch)
    plan = ModelPlan.from_settings(settings)
    assert plan.pool and plan.conversation()[0].model_name == "sol"
    workspace = Workspace(tmp_path, "base", "head")
    simplify = simplifier_agents(plan)(workspace)
    pairs = [correctness_agents(plan)(workspace), *(lens.agents(workspace) for lens in build_lenses(settings, plan))]
    for finder, check in [*pairs, (simplify.propose, simplify.check)]:
        assert finder.model and finder.model.model_name in {"a", "b"}
        checker = check()
        assert checker.model and checker.model.model_name == "sol"
    assert models_note(settings) == "found by model pool: openai:a, openai:b; verified by `openai:sol`"


def test_an_overflow_finder_on_the_checker_is_checked_by_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = settings_for(tmp_path, monkeypatch, overflow=CHECKER)
    plan = ModelPlan.from_settings(settings)
    select = Mock()
    overflow = overflow_deps(settings, deps(tmp_path, select), plan)
    assert (
        overflow
        and overflow.models_note == "found by `openai:sol`, checks scheduled from model pool: openai:a, openai:b"
    )
    workspace = Workspace(tmp_path, "base", "head")
    assert overflow.simplifier
    simplify = overflow.simplifier(workspace)
    pairs = [overflow.agents(workspace), *(lens.agents(workspace) for lens in overflow.lenses)]
    for finder, check in [*pairs, (simplify.propose, simplify.check)]:
        assert finder.model and finder.model.model_name == "sol"
        checker = check()
        assert checker.model and checker.model.model_name in {"a", "b"}
    select.assert_not_called()


def test_another_overflow_finder_keeps_the_dedicated_checker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_for(tmp_path, monkeypatch, overflow="openai:burst")
    overflow = overflow_deps(settings, deps(tmp_path), ModelPlan.from_settings(settings))
    assert overflow and overflow.models_note == "found by `openai:burst`, verified by `openai:sol`"
    finder, check = overflow.agents(Workspace(tmp_path, "base", "head"))
    assert finder.model and finder.model.model_name == "burst"
    checker = check()
    assert checker.model and checker.model.model_name == "sol"


@pytest.mark.parametrize(
    ("options", "error"),
    [
        pytest.param({"pool": POOL, "checks": "checker"}, "explicit checker", id="no checker"),
        pytest.param(
            {"finder": "openai:a", "checker": CHECKER, "checks": "checker"}, "only applies to a pool", id="no pool"
        ),
    ],
)
def test_a_dedicated_checker_needs_a_pool_and_a_checker(options: dict[str, object], error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        PrismConfig.model_validate(options)


def test_cli_writes_a_dedicated_checker(tmp_path: Path) -> None:
    path = tmp_path / "prism.json"
    main(
        [
            *("--output", str(path), "--checker", CHECKER, "--checks", "checker"),
            *("--pool-model", "openai:a=1", "--pool-model", "openai:b=2"),
        ]
    )
    config = PrismConfig.from_file(path)
    assert (config.checks, config.checker, config.finder, config.pool) == ("checker", CHECKER, "openai:a", POOL)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        pytest.param(["--checker", CHECKER, "--checks", "checker"], "--checks requires --pool-model", id="no pool"),
        pytest.param(
            ["--pool-model", "openai:a=1", "--pool-model", "openai:b=2", "--checks", "checker"],
            "--checks checker requires --checker",
            id="no checker",
        ),
    ],
)
def test_cli_rejects_checks_without_their_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str], error: str
) -> None:
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "prism.json"), "--finder", "openai:a", *args])
    assert error in capsys.readouterr().err
