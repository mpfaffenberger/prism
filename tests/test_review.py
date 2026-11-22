"""The review: git-derived diff facts, the checkout, the agents' git tools, the pipeline, and the job's policies."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from autoreviewer.review.agents import correctness_agent
from autoreviewer.review.diff import (
    ChangedPath,
    annotate_git_diff,
    commentable_lines,
    parse_changes,
    select,
    split_git_diff,
)
from autoreviewer.review.gittools import git_tools
from autoreviewer.review.job import _summary, finish_check, review_and_post, start_check
from autoreviewer.review.pipeline import OUT_OF_TIME, Outcome, PullRequestText, lens_prompt, review
from autoreviewer.review.posting import VERDICT_MARKER
from autoreviewer.review.render import MAX_PATH, code_span, comment_body, location, review_body, sanitize
from autoreviewer.review.schemas import Finding
from autoreviewer.review.workspace import BASE, HEAD, Workspace, read_changes, run_git
from autoreviewer.web.routing import PullRequestUpdated
from tests.review_fakes import BOT, HEAD_SOURCE, FakeGitHub, GitRepo, add_commit, agents, finding, job_deps
from tests.test_withdrawals import withdraw

pytestmark = pytest.mark.anyio


def pull(repo: GitRepo) -> PullRequestUpdated:
    return PullRequestUpdated(installation_id=1, owner="mpfaffenberger", repo="demo", number=7, head_sha=repo.head)


# --- diff facts (pure) ----------------------------------------------------------------------------


def test_commentable_lines_are_the_new_side_of_the_hunks() -> None:
    patch = "@@ -10,4 +20,5 @@ def f():\n a\n-b\n+c\n+d\n e\n\\ No newline at end of file"
    assert commentable_lines(patch) == frozenset({20, 21, 22, 23})


def test_parse_changes_handles_renames_deletions_and_binaries() -> None:
    name_status = "M\0a.py\0R087\0old.py\0new.py\0D\0gone.py\0A\0logo.png\0"
    # Joined with "\0".join, not adjacent literals: a formatter merging "...\0" "2..." yields the octal escape "\02".
    numstat = "\0".join(["3\t1\ta.py", "2\t2\t", "old.py", "new.py", "0\t5\tgone.py", "-\t-\tlogo.png", ""])
    assert parse_changes(name_status, numstat) == [
        ChangedPath("a.py", "M", 3, 1),
        ChangedPath("new.py", "R", 2, 2, old_path="old.py"),
        ChangedPath("gone.py", "D", 0, 5),
        ChangedPath("logo.png", "A", None, None),
    ]


def test_select_skips_generated_vendored_deleted_and_binary_files() -> None:
    changes = [
        ChangedPath("src/app.py", "M", 1, 1),
        ChangedPath("uv.lock", "M", 9, 9),
        ChangedPath("web/vendor/lib.js", "A", 5, 0),
        ChangedPath("old.py", "D", 0, 4),
        ChangedPath("logo.png", "A", None, None),
    ]
    selection = select(changes)
    assert [c.path for c in selection.reviewable] == ["src/app.py"]
    assert selection.changed_lines == 2


def test_split_and_annotate_a_multi_file_git_diff() -> None:
    text = (
        "diff --git a/x.py b/x.py\nindex 1..2 100644\n--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a\n-b\n+c\n"
        "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    )
    assert split_git_diff(text) == {"x.py": "@@ -1,2 +1,2 @@\n a\n-b\n+c"}
    annotated = annotate_git_diff(text).splitlines()
    assert annotated[3] == "       +++ b/x.py"  # header lines get no number, even though they start with "-"/"+"
    assert annotated[5] == "     1  a" and annotated[6] == "       -b" and annotated[7] == "     2 +c"


# --- checkout -------------------------------------------------------------------------------------


async def test_checkout_has_head_files_and_named_refs(repo: GitRepo, workspace: Workspace) -> None:
    assert (workspace.root / "stats.py").read_text().endswith("- 1\n")
    assert (await run_git(workspace.root, "rev-parse", BASE)).strip() == repo.base
    assert (await run_git(workspace.root, "rev-parse", HEAD)).strip() == repo.head


async def test_symlinks_are_checked_out_as_plain_files(workspace: Workspace) -> None:
    link = workspace.root / "secrets"
    assert not link.is_symlink() and link.read_text() == "/etc/passwd"  # the target's name, not its contents


async def test_read_changes_lists_files_and_patches(workspace: Workspace) -> None:
    changes, patches = await read_changes(workspace)
    assert {(c.path, c.status) for c in changes} == {("stats.py", "M"), ("uv.lock", "M"), ("notes.py", "A")}
    assert 5 in commentable_lines(patches["stats.py"])


# --- the agents' git tools ------------------------------------------------------------------------


async def call(workspace: Workspace, name: str, **args: object) -> str:
    tool = (git_tools(workspace).tools)[name]
    return await tool.function(**args)  # type: ignore[misc]


async def test_changed_files_marks_what_is_not_reviewed(workspace: Workspace) -> None:
    listing = await call(workspace, "changed_files")
    assert "M +1 -1 stats.py" in listing
    assert "uv.lock  (generated, vendored, binary, or deleted: not reviewed)" in listing


async def test_diff_is_annotated_with_new_line_numbers(workspace: Workspace) -> None:
    text = await call(workspace, "diff", path="stats.py")
    assert "     5 +    return total / len(xs) - 1" in text


async def test_diff_pages_large_output(workspace: Workspace) -> None:
    first = await call(workspace, "diff", limit=3)
    assert first.startswith("[diff (all files): lines 1-3 of") and "call again with offset=3" in first


async def test_show_file_gives_the_version_before_the_pr(workspace: Workspace) -> None:
    text = await call(workspace, "show_file", path="stats.py", ref="base")
    assert "     5      return total / len(xs)" in text and "- 1" not in text
    assert await call(workspace, "show_file", path="notes.py", ref="base") == "notes.py does not exist at base"


@pytest.mark.parametrize("bad", ["/run/prism/github-app.pem", "../outside.py", "--output=/tmp/x", "a\0b", ""])
async def test_git_tools_refuse_paths_outside_the_repo(workspace: Workspace, bad: str) -> None:
    with pytest.raises(ModelRetry):
        await call(workspace, "diff", path=bad)


# --- render ---------------------------------------------------------------------------------------


def test_sanitize_neutralizes_mentions_but_not_emails_or_code() -> None:
    text = sanitize("ping @mpfaffenberger and @org/team, mail a@b.com, code `@decorator`")
    assert "@\u200bmpfaffenberger" in text and "@\u200borg/team" in text
    assert "a@b.com" in text and "`@decorator`" in text


def test_sanitize_redacts_secrets_and_caps_length() -> None:
    text = sanitize("token ghp_" + "a" * 36 + " key sk_bood_ABCDEFGHIJ " + "x" * 5000)
    assert "ghp_" not in text and "sk_bood_" not in text and text.count("[redacted]") == 2
    assert len(text) <= 3000


def test_comment_body_runs_through_sanitize() -> None:
    body = comment_body(Finding.model_validate({**finding(), "explanation": "cc @someone"}))
    assert "@\u200bsomeone" in body and "**[P1] Mean is off by one** (correctness)" in body


# --- pipeline -------------------------------------------------------------------------------------


async def pr_text(workspace: Workspace) -> PullRequestText:
    changes, patches = await read_changes(workspace)
    selection = select(changes)
    return PullRequestText("Add average()", "", selection.reviewable, selection.skipped, patches)


async def test_the_lens_prompt_lists_files_but_carries_no_diff(workspace: Workspace) -> None:
    prompt = lens_prompt(await pr_text(workspace))
    assert "- M stats.py (+1 -1)" in prompt and "1 generated, vendored, binary, or deleted" in prompt
    assert "len(xs) - 1" not in prompt  # the model reads the change through its tools


async def test_confirmed_findings_on_diff_lines_are_inline(workspace: Workspace) -> None:
    lens, verifier = agents([finding(5)])(workspace)
    outcome = await review(await pr_text(workspace), lens, verifier)
    assert [(f.path, f.line) for f in outcome.inline] == [("stats.py", 5)]
    assert outcome.general == [] and outcome.blocking


async def test_verifier_rejections_are_dropped(workspace: Workspace) -> None:
    lens, verifier = agents([finding(5), finding(3, title="Loop is wrong")], confirm=lambda f: f["line"] == 5)(
        workspace
    )
    outcome = await review(await pr_text(workspace), lens, verifier)
    assert [f.line for f in outcome.inline] == [5]
    assert [(f.line, reason) for f, reason in outcome.rejected] == [(3, "not a real bug")]


async def test_findings_off_the_diff_go_to_the_body_and_unchanged_paths_are_dropped(workspace: Workspace) -> None:
    lens, verifier = agents([finding(40), finding(1, path="uv.lock"), finding(1, path="never_touched.py")])(workspace)
    outcome = await review(await pr_text(workspace), lens, verifier)
    assert [f.line for f in outcome.general] == [40]
    assert [reason for _, reason in outcome.rejected] == ["path is not among the changed files"] * 2


async def test_duplicate_findings_are_reported_once(workspace: Workspace) -> None:
    lens, verifier = agents([finding(5), finding(5, title="Same bug, other words")])(workspace)
    assert len((await review(await pr_text(workspace), lens, verifier)).inline) == 1


# --- job (the review_and_post phase; composition and recovery are in test_durable.py) --------------


async def test_a_bug_requests_changes_with_an_inline_comment(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    result = await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo))

    [posted] = github.reviews
    assert posted["event"] == "REQUEST_CHANGES" and posted["commit_sha"] == repo.head
    assert [(c.path, c.line) for c in github.inline_comments] == [("stats.py", 5)]
    assert (result.conclusion, result.title) == ("failure", "1 blocking issue(s)")


@pytest.mark.parametrize(
    ("name", "span"),
    [
        ("stats.py:5", "`stats.py:5`"),
        ("a`b.py:5", "``a`b.py:5``"),
        ("`x``y`:5", "``` `x``y`:5 ```"),  # padded on both sides: Markdown strips one space from each
        ("end`", "`` end` ``"),
        ("two\nlines.py:5", "`two lines.py:5`"),
        ("tab\t\r\nhere.py:5", "`tab here.py:5`"),
        ("two  spaces.py:5", "`two  spaces.py:5`"),  # a file's own spaces are part of its name
    ],
)
def test_a_file_name_cannot_close_its_code_span(name: str, span: str) -> None:
    """From the bot's review of #65: the PR's author picks the file names the bot's review then prints."""
    assert code_span(name) == span


def test_a_long_path_is_clipped_before_its_line_number() -> None:
    """From the bot's review of #65: clipping the whole location lost the line."""
    assert location("stats.py", 5) == "`stats.py:5`"
    assert location("d/" * 300 + "stats.py", 5) == f"`{('d/' * 300)[: MAX_PATH - 1]}…:5`"


def test_an_unchecked_candidate_on_a_hostile_file_name_stays_inside_its_span() -> None:
    hostile = "x`[approve](https://evil.example) @octocat `.py"
    unchecked = Finding.model_validate({**finding(5, path=hostile), "kind": "correctness"})
    body = review_body(Outcome(unverified=[(unchecked, OUT_OF_TIME)], cut_short=True), footer="footer")
    assert f"``{sanitize(hostile)}:5``" in body and "@octocat" not in body  # no ping, and the span holds


def test_a_finding_outside_the_diff_on_a_hostile_file_name_stays_inside_its_span() -> None:
    """From the bot's review of #65: the list of findings outside the changed lines prints file names too."""
    hostile = "x`[approve](https://evil.example) @octocat `.py"
    outside = Finding.model_validate({**finding(5, path=hostile), "kind": "correctness"})
    body = review_body(Outcome(general=[outside]), footer="footer")
    assert f"- ``{sanitize(hostile)}:5``: " in body and "@octocat" not in body


async def test_a_clean_pr_is_approved(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    result = await review_and_post(job_deps(github, tmp_path, []), pull(repo))
    assert github.reviews[0]["event"] == "APPROVE" and github.reviews[0]["comments"] == []
    assert result.conclusion == "success"


@pytest.mark.parametrize("hung", ["settle_fixed", "_publish"])
async def test_expiring_after_a_blocking_verdict_preserves_the_failed_check(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hung: str
) -> None:
    github = FakeGitHub(repo)

    async def hang(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    cleanup = AsyncMock(side_effect=hang)
    monkeypatch.setattr(f"autoreviewer.review.job.{hung}", cleanup)
    deps = replace(job_deps(github, tmp_path, [finding(5)]), hard_deadline_s=1.5)  # the hung cleanup outlasts it
    async with asyncio.timeout(5):
        result = await review_and_post(deps, pull(repo), 1)

    cleanup.assert_awaited_once()
    assert [r["event"] for r in github.reviews] == ["REQUEST_CHANGES"]
    assert (result.conclusion, result.title) == ("failure", "1 blocking issue(s)")


async def test_rejected_inline_comments_fall_back_to_the_body(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo, reject_inline=True)
    result = await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo))
    [posted] = github.reviews
    assert posted["event"] == "REQUEST_CHANGES" and posted["comments"] == []
    assert "`stats.py:5`" in posted["body"] and result.conclusion == "failure"


async def test_a_commit_already_reviewed_is_not_reviewed_again(repo: GitRepo, tmp_path: Path) -> None:
    # What a replay looks like after a crash between posting the review and checkpointing the step.
    github = FakeGitHub(repo)
    verdict = f"Found 1 issue.\n\n{VERDICT_MARKER}"
    github.reviews.append(
        {"commit_sha": repo.head, "event": "REQUEST_CHANGES", "body": verdict, "comments": [], "author": BOT}
    )
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo))
    assert len(github.reviews) == 1


async def test_same_head_attempt_posts_its_own_verdict_but_replay_does_not(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    event = pull(repo)
    clean = job_deps(github, tmp_path)
    blocking = job_deps(github, tmp_path, [finding(5)])
    await review_and_post(clean, event, check_id=1)
    for deps, check_id in ((blocking, 2), (clean, 2), (blocking, 3)):
        # A replay must retain the published REQUEST_CHANGES even if a second model run would approve.
        result = await review_and_post(deps, event, check_id=check_id)
        assert result.conclusion == "failure"
    assert [r["event"] for r in github.reviews] == ["APPROVE", "REQUEST_CHANGES", "REQUEST_CHANGES"]
    assert len(github.review_comments) == 1


async def test_expired_recovery_finishes_check_from_published_verdict(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    event = pull(repo)
    deps = job_deps(github, tmp_path, [finding(5)])
    check = await start_check(deps, event)
    await review_and_post(deps, event, check)
    # Lost the step checkpoint and stayed down beyond the hard deadline.
    await finish_check(deps, event, check, await review_and_post(deps, event, check, remaining_s=0))
    assert github.checks[-1]["conclusion"] == "failure"
    fresh = await start_check(deps, event)
    await finish_check(deps, event, fresh, await review_and_post(deps, event, fresh, remaining_s=0))
    assert github.checks[-1]["title"] == "Review timed out"  # cannot borrow the previous check's verdict
    assert len(github.reviews) == 1


@pytest.mark.parametrize(
    ("soft_deadline_s", "conclusion", "title"),
    [(540.0, "success", "No blocking issues"), (0.0, "neutral", "Partially reviewed")],
    ids=["approval", "partial"],
)
async def test_expired_recovery_reports_a_published_approval_or_partial(
    repo: GitRepo, tmp_path: Path, soft_deadline_s: float, conclusion: str, title: str
) -> None:
    """From the bot's review of #61: only a recovered REQUEST_CHANGES was covered."""
    github = FakeGitHub(repo)
    event = pull(repo)
    deps = replace(job_deps(github, tmp_path), soft_deadline_s=soft_deadline_s)
    check = await start_check(deps, event)
    await review_and_post(deps, event, check)
    await finish_check(deps, event, check, await review_and_post(deps, event, check, remaining_s=0))
    assert (github.checks[-1]["conclusion"], github.checks[-1]["title"]) == (conclusion, title)
    assert len(github.reviews) == 1


async def test_asking_again_cannot_approve_over_a_standing_finding(repo: GitRepo, tmp_path: Path) -> None:
    """From the bot's review of #61: a rerun on unchanged code that misses a confirmed blocker must not turn
    the commit green, or asking until one pass misses it would clear any finding."""
    github = FakeGitHub(repo)
    event = pull(repo)
    blocked = await review_and_post(job_deps(github, tmp_path, [finding(5)]), event, check_id=1)
    again = await review_and_post(job_deps(github, tmp_path), event, check_id=2)  # this pass finds nothing
    assert (blocked.conclusion, again.conclusion) == ("failure", "failure")
    assert again.title == "Earlier blocking finding stands"
    assert [r["event"] for r in github.reviews] == ["REQUEST_CHANGES", "REQUEST_CHANGES"]
    assert "**Still blocked:**" in github.reviews[-1]["body"] and "No new issues found." in github.reviews[-1]["body"]
    assert len(github.review_comments) == 1  # the standing comment is not posted again


async def test_a_finding_withdrawn_in_its_thread_no_longer_stands(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    event = pull(repo)
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), event, check_id=1)
    withdraw(github, 1)
    again = await review_and_post(job_deps(github, tmp_path), event, check_id=2)
    assert again.conclusion == "success" and github.reviews[-1]["event"] == "APPROVE"


async def test_a_blocker_listed_without_a_thread_stands_too(repo: GitRepo, tmp_path: Path) -> None:
    """Its verdict counted it as `outside`: nobody can withdraw it in a thread, so only a new commit clears it."""
    github = FakeGitHub(repo, fail_review_comments=True, reject_inline=True)
    event = pull(repo)
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), event, check_id=1)
    assert "outside=1" in github.reviews[-1]["body"] and github.review_comments == []
    again = await review_and_post(job_deps(github, tmp_path), event, check_id=2)
    assert again.conclusion == "failure" and github.reviews[-1]["event"] == "REQUEST_CHANGES"


async def test_a_new_commit_is_judged_on_its_own(repo: GitRepo, tmp_path: Path) -> None:
    """The earlier commit's comment still shows on its line, but it is not a finding against this code."""
    github = FakeGitHub(repo)
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo), check_id=1)
    github.repo = replace(repo, head=add_commit(repo, {"stats.py": HEAD_SOURCE.replace(" - 1", "")}))
    fixed = await review_and_post(job_deps(github, tmp_path), pull(github.repo), check_id=2)
    assert fixed.conclusion == "success" and github.reviews[-1]["event"] == "APPROVE"


async def test_no_approval_when_standing_findings_cannot_be_read(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo, fail_posted_comments=2)  # the withdrawals read, then the standing-findings read
    result = await review_and_post(job_deps(github, tmp_path), pull(repo), check_id=1)
    assert (result.title, result.conclusion) == ("Review failed", "neutral") and github.reviews == []


async def test_timeout_finishes_even_if_review_lookup_fails(
    repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    github = FakeGitHub(repo)
    event = pull(repo)
    deps = job_deps(github, tmp_path)
    check = await start_check(deps, event)
    monkeypatch.setattr(github, "review_bodies", AsyncMock(side_effect=OSError("unavailable")))
    await finish_check(deps, event, check, await review_and_post(deps, event, check, remaining_s=0))
    assert github.checks[-1]["status"] == "completed" and github.checks[-1]["conclusion"] == "neutral"


async def test_early_comments_do_not_count_as_the_verdict(repo: GitRepo, tmp_path: Path) -> None:
    """GitHub wraps each early comment in an empty COMMENTED review; the verdict must still be posted (live on #21)."""
    github = FakeGitHub(repo)
    await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo))
    assert len(github.review_comments) == 1 and [r["event"] for r in github.reviews] == ["REQUEST_CHANGES"]


async def test_a_push_during_the_review_supersedes_it(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo, head_shas=[repo.head, "b" * 40])  # moved on between the start and the post
    result = await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo))
    assert github.reviews == []
    assert (result.title, result.conclusion) == ("Superseded", "neutral")


async def test_huge_prs_are_not_reviewed(repo: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autoreviewer.review.job.MAX_CHANGED_LINES", 1)
    github = FakeGitHub(repo)
    result = await review_and_post(job_deps(github, tmp_path, [finding(5)]), pull(repo))
    assert github.reviews == [] and result.title == "Too large to review"


async def test_a_crash_becomes_a_result_instead_of_an_exception(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo, head_shas=[repo.head])
    deps = job_deps(github, tmp_path, [finding(5)])

    async def broken_compare(*_: object) -> object:
        raise RuntimeError("GitHub is down")

    github.compare = broken_compare  # type: ignore[method-assign]
    result = await review_and_post(deps, pull(repo))
    assert github.reviews == [] and (result.title, result.conclusion) == ("Review failed", "neutral")


# --- tool retries ---------------------------------------------------------------------------------


async def test_a_few_misses_on_missing_paths_do_not_end_the_review(workspace: Workspace) -> None:
    """Regression for PR #5: four list_directory calls on a missing path killed an 8-minute review."""
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from autoreviewer.review.agents import LENS_LIMITS, correctness_agent

    misses = iter(range(6))

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if next(misses, None) is not None:
            return ModelResponse(parts=[ToolCallPart(tool_name="list_directory", args={"path": ".cache"})])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=info.output_tools[0].name, args={"findings": [], "overall_correct": True, "reviewed": []}
                )
            ]
        )

    result = await correctness_agent(FunctionModel(respond), workspace).run("Review.", usage_limits=LENS_LIMITS)
    assert result.output.findings == []


# --- priorities, confidence, and blocking (lessons from Codex and PR-Agent prompts) ----------------


async def test_advisory_findings_are_posted_but_approve(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    result = await review_and_post(job_deps(github, tmp_path, [finding(5, priority=2)]), pull(repo))
    [posted] = github.reviews
    assert posted["event"] == "APPROVE" and len(github.inline_comments) == 1
    assert "(correctness, non-blocking)" in github.inline_comments[0].body
    assert result.conclusion == "success" and "1 advisory (non-blocking) finding(s) posted." in result.summary


async def test_the_verifier_can_raise_a_findings_priority(repo: GitRepo, tmp_path: Path) -> None:
    github = FakeGitHub(repo)
    deps = job_deps(github, tmp_path, [finding(5, priority=3)], priority=lambda f: 0)
    await review_and_post(deps, pull(repo))
    assert github.reviews[0]["event"] == "REQUEST_CHANGES"
    assert github.inline_comments[0].body.startswith("**[P0] ")


async def test_low_confidence_findings_never_reach_the_verifier(workspace: Workspace) -> None:
    asked: list[int] = []

    def confirm(reported: dict[str, object]) -> bool:
        asked.append(int(reported["line"]))  # type: ignore[arg-type]
        return True

    lens, verifier = agents([finding(5, confidence=9), finding(3, title="Hunch", confidence=2)], confirm=confirm)(
        workspace
    )
    outcome = await review(await pr_text(workspace), lens, verifier)
    assert asked == [5]
    assert [f.line for f in outcome.low_confidence] == [3] and not outcome.rejected
    assert "dropped by verification" not in _summary(outcome).summary


async def test_dedupe_keeps_the_most_urgent_finding_on_a_line(workspace: Workspace) -> None:
    lens, verifier = agents([finding(5, priority=2, title="Minor"), finding(5, priority=0, title="Major")])(workspace)
    outcome = await review(await pr_text(workspace), lens, verifier)
    assert [(f.priority, f.title) for f in outcome.posted] == [(0, "Major")]


@pytest.mark.parametrize(("block_at", "event"), [(1, "APPROVE"), (3, "REQUEST_CHANGES")])
async def test_block_at_priority_is_configurable(repo: GitRepo, tmp_path: Path, block_at: int, event: str) -> None:
    from dataclasses import replace

    github = FakeGitHub(repo)
    deps = replace(job_deps(github, tmp_path, [finding(5, priority=3)]), block_at=block_at)
    await review_and_post(deps, pull(repo))
    assert github.reviews[0]["event"] == event


async def test_a_report_split_across_two_output_calls_keeps_the_findings(workspace: Workspace) -> None:
    """Qwen sometimes sends the verdict in one final_result call and the findings in a second one. The
    first, partial call must not validate on its own, or its empty default would win and drop every bug."""

    def split(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name = info.output_tools[0].name
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name=name, args={"overall_correct": False, "reviewed": ["stats.py"]}),
                ToolCallPart(tool_name=name, args={"findings": [finding(5)], "overall_correct": False, "reviewed": []}),
            ]
        )

    _, verifier = agents([])(workspace)
    outcome = await review(await pr_text(workspace), correctness_agent(FunctionModel(split), workspace), verifier)
    assert [f.line for f in outcome.posted] == [5]


async def test_the_verifier_corrects_a_wrong_detail_instead_of_dropping_a_real_bug(workspace: Workspace) -> None:
    lens, verifier = agents([finding(5)], explanation=lambda f: "Corrected: the existing test does catch it.")(
        workspace
    )
    outcome = await review(await pr_text(workspace), lens, verifier)
    assert [f.explanation for f in outcome.posted] == ["Corrected: the existing test does catch it."]
