"""On-demand reviews of arbitrary PRs: parsing the reference, the review itself, and read-only GitHub access."""

from __future__ import annotations

from pathlib import Path

import httpx2
import pytest

from autoreviewer import ondemand
from autoreviewer.github import PublicGitHub, Repo
from autoreviewer.ondemand import PullRef, parse_pr_ref, review_pr
from tests.review_fakes import FakeGitHub, GitRepo, LocalSource, agents, finding

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://github.com/pydantic/pydantic-ai/pull/1234", PullRef("pydantic", "pydantic-ai", 1234)),
        ("https://github.com/pydantic/pydantic-ai/pull/1234/files", PullRef("pydantic", "pydantic-ai", 1234)),
        ("https://github.com/o/r.x/pull/7/commits/abc123?w=1#diff", PullRef("o", "r.x", 7)),
        ("  mpfaffenberger/code_puppy#42 ", PullRef("mpfaffenberger", "code_puppy", 42)),
    ],
)
def test_pr_references_are_parsed(text: str, expected: PullRef) -> None:
    assert parse_pr_ref(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "http://github.com/o/r/pull/1",
        "https://gitlab.com/o/r/pull/1",
        "https://github.com/o/r/issues/1",
        "https://github.com/o/r/pull/",
        "https://github.com/o/r/pull/1; rm -rf /",
        "o/r#1 extra",
        "../r#1",
        "",
    ],
)
def test_anything_else_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="not a GitHub pull request"):
        parse_pr_ref(text)


REF = PullRef("someone", "elsewhere", 7)


async def test_a_buggy_pr_reports_request_changes_and_writes_nothing(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    report = await review_pr(
        REF, reader=github, source=LocalSource(repo), agents=agents([finding(5)]), work_dir=tmp_path
    )

    assert report.exit_code == ondemand.EXIT_REQUEST_CHANGES and report.head_sha == repo.head
    text = report.render()
    assert "**Verdict: REQUEST CHANGES** (nothing was posted to GitHub)" in text
    assert "## `stats.py:5`" in text and "Mean is off by one" in text
    assert github.reviews == [] and github.checks == []


async def test_a_clean_pr_reports_approve(repo: GitRepo, tmp_path: Path) -> None:
    report = await review_pr(
        REF, reader=FakeGitHub(repo), source=LocalSource(repo), agents=agents([]), work_dir=tmp_path
    )
    assert report.exit_code == ondemand.EXIT_APPROVE and "**Verdict: APPROVE**" in report.render()


async def test_dropped_candidates_are_listed_with_the_verifiers_reason(repo: GitRepo, tmp_path: Path) -> None:
    report = await review_pr(
        REF,
        reader=FakeGitHub(repo),
        source=LocalSource(repo),
        agents=agents([finding(5), finding(3, title="Loop is wrong")], confirm=lambda f: f["line"] == 5),
        work_dir=tmp_path,
    )
    assert "1 candidate(s) dropped by verification" in report.render()
    assert "`stats.py:3` Loop is wrong: not a real bug" in report.render()


async def test_too_large_prs_say_so(repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autoreviewer.review.job.MAX_CHANGED_LINES", 1)
    report = await review_pr(
        REF, reader=FakeGitHub(repo), source=LocalSource(repo), agents=agents([]), work_dir=tmp_path
    )
    assert report.exit_code == ondemand.EXIT_NOT_REVIEWED and "**Too large to review**" in report.render()


@pytest.mark.parametrize(("block_at", "exit_code"), [(1, ondemand.EXIT_APPROVE), (3, ondemand.EXIT_REQUEST_CHANGES)])
async def test_the_cli_verdict_honours_block_at(repo: GitRepo, tmp_path: Path, block_at: int, exit_code: int) -> None:
    report = await review_pr(
        REF, reader=FakeGitHub(repo), source=LocalSource(repo), agents=agents([finding(5, priority=3)]),
        work_dir=tmp_path, block_at=block_at,
    )  # fmt: skip
    assert report.exit_code == exit_code


@pytest.mark.parametrize(("token", "expected_auth"), [(None, None), ("ghp_example", "token ghp_example")])
async def test_public_github_sends_credentials_only_when_given_a_token(
    token: str | None, expected_auth: str | None
) -> None:
    seen: list[str | None] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request.headers.get("authorization"))
        return httpx2.Response(200, json={"merge_base_commit": {"sha": "c" * 40}, "status": "ahead"})

    github = PublicGitHub(token)
    github._github = type(github._github)(github._github.auth, async_transport=httpx2.MockTransport(handle))
    assert (await github.compare(Repo(0, "someone", "elsewhere"), "b" * 40, "a" * 40)).merge_base == "c" * 40
    assert [value.lower() if value else None for value in seen] == [expected_auth.lower() if expected_auth else None]


# --- exit codes: 0 approve, 1 request changes, 2 error (a crash must never look like a verdict) ------


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from autoreviewer.web.config import Settings

    settings = Settings(
        app_id=1,
        app_slug="s",
        webhook_secret="x",
        private_key="k",
        boodleton_api_key="test-key",
        work_dir=tmp_path,
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls, env=None: settings))


def _report(result: object) -> object:
    from autoreviewer.ondemand import Report

    return Report(REF, "a" * 40, "title", result)  # type: ignore[arg-type]


def test_the_report_tells_unchecked_candidates_from_verifier_failures() -> None:
    """From the bot's review of #65: a candidate the time limit left unchecked is not a verifier failure."""
    from autoreviewer.review.pipeline import OUT_OF_TIME, Outcome
    from autoreviewer.review.schemas import Finding

    late = Finding.model_validate(finding(5, title="Not reached in time"))
    broken = Finding.model_validate(finding(4, title="Verifier crashed on this"))
    outcome = Outcome(unverified=[(late, OUT_OF_TIME), (broken, "verification failed: RuntimeError")], cut_short=True)
    text = ondemand.Report(REF, "a" * 40, "title", outcome).render()
    unchecked, failed = text.split("could not be verified")
    assert "**1 candidate(s) were not checked before the time limit**" in unchecked
    assert "- `stats.py:5`\n" in unchecked and "Verifier crashed" not in unchecked
    assert "`stats.py:4` Verifier crashed on this: verification failed: RuntimeError" in failed
    assert "Not reached in time" not in text and "PARTIALLY REVIEWED" in text  # where, never the unverified claim


@pytest.mark.usefixtures("cli")
@pytest.mark.parametrize(("blocking", "code"), [(True, 1), (False, 0)])
def test_verdicts_map_to_exit_codes(monkeypatch: pytest.MonkeyPatch, blocking: bool, code: int) -> None:
    from autoreviewer import ondemand
    from autoreviewer.review.pipeline import Outcome
    from autoreviewer.review.schemas import Finding

    outcome = Outcome(inline=[Finding.model_validate(finding(5))] if blocking else [])

    async def fake_review(*_: object, **__: object) -> object:
        return _report(outcome)

    monkeypatch.setattr(ondemand, "review_pr", fake_review)
    assert ondemand.run(["o/r#1"]) == code


@pytest.mark.usefixtures("cli")
def test_the_cli_reviews_under_the_bots_time_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """From the bot's review of #65: without the checking cutoff an on-demand review waits for every check and
    can hit the hard stop with no verdict."""
    from autoreviewer.review.deadline import DEFAULT_SOFT_S, MAX_REVIEW_S, Deadline, checking_cutoff
    from autoreviewer.review.pipeline import Outcome

    seen: list[Deadline] = []

    async def fake_review(*_: object, deadline: Deadline, **__: object) -> object:
        seen.append(deadline)
        return _report(Outcome())

    monkeypatch.setattr(ondemand, "review_pr", fake_review)
    assert ondemand.run(["o/r#1"]) == 0
    [deadline] = seen
    assert (deadline.soft_s, deadline.checking_s) == (DEFAULT_SOFT_S, checking_cutoff(MAX_REVIEW_S))


@pytest.mark.usefixtures("cli")
def test_a_crash_exits_2_not_1(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from autoreviewer import ondemand

    async def boom(*_: object, **__: object) -> object:
        raise RuntimeError("GitHub is down")

    monkeypatch.setattr(ondemand, "review_pr", boom)
    assert ondemand.run(["o/r#1"]) == ondemand.EXIT_ERROR
    assert "error: could not review o/r#1: RuntimeError: GitHub is down" in capsys.readouterr().err


def test_missing_configuration_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    from autoreviewer import ondemand
    from autoreviewer.web.config import Settings

    def broken(cls: type, env: object = None) -> Settings:
        raise KeyError("BOODLETON_API_KEY")

    monkeypatch.setattr(Settings, "from_env", classmethod(broken))
    assert ondemand.run(["o/r#1"]) == ondemand.EXIT_ERROR


def test_a_bad_reference_exits_2() -> None:
    from autoreviewer import ondemand

    with pytest.raises(SystemExit) as exit_info:
        ondemand.run(["not-a-pr"])
    assert exit_info.value.code == ondemand.EXIT_ERROR
