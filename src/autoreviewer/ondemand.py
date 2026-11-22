"""Review any GitHub PR on demand, including repos where the App isn't installed. Nothing is posted.

    podman exec prism autoreviewer-review-pr https://github.com/owner/repo/pull/123
    podman exec -e GITHUB_TOKEN=... prism autoreviewer-review-pr owner/repo#123   # private repos

Runs the bot's own pipeline (git workspace, lens, verifier, compaction, limits) inside the bot's
hardened container, and prints the verdict the bot would give.

Exit codes: 0 would approve (or nothing to review), 1 would request changes, 2 error, 3 not reviewed
(e.g. too large). Errors never exit 1, so a crash can't be mistaken for a blocking verdict (caught by the
bot reviewing this file), and a refusal never exits 0, so it can't be mistaken for an approval.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from autoreviewer.github import PublicGitHub, PullRequestReader, Repo
from autoreviewer.review.deadline import Deadline, checking_cutoff
from autoreviewer.review.job import AgentFactory, CheckResult, Lens, Reviewed, review_change
from autoreviewer.review.pipeline import BLOCK_AT_PRIORITY, Outcome
from autoreviewer.review.render import comment_body, location
from autoreviewer.review.simplify import SimplifierFactory
from autoreviewer.review.workspace import PublicSource, RepoSource

logger = logging.getLogger("autoreviewer.ondemand")

EXIT_APPROVE, EXIT_REQUEST_CHANGES, EXIT_ERROR, EXIT_NOT_REVIEWED = 0, 1, 2, 3

_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})"
_URL = re.compile(rf"^https://github\.com/({_NAME})/({_NAME})/pull/(\d+)(?:/[A-Za-z0-9/_-]*)?(?:[?#].*)?$")
_SHORT = re.compile(rf"^({_NAME})/({_NAME})#(\d+)$")


@dataclass(frozen=True)
class PullRef:
    owner: str
    repo: str
    number: int


def parse_pr_ref(text: str) -> PullRef:
    """Accept https://github.com/owner/repo/pull/N (any suffix like /files) or owner/repo#N."""
    text = text.strip()
    if (match := _URL.match(text) or _SHORT.match(text)) is None:
        raise ValueError(f"not a GitHub pull request: {text!r} (expected https://github.com/owner/repo/pull/N)")
    owner, repo, number = match.groups()
    return PullRef(owner, repo.removesuffix(".git"), int(number))


@dataclass(frozen=True)
class Report:
    ref: PullRef
    head_sha: str
    title: str
    result: Outcome | CheckResult

    @property
    def exit_code(self) -> int:
        if isinstance(self.result, CheckResult):  # no review ran; "success" means there was nothing to review
            return EXIT_APPROVE if self.result.conclusion == "success" else EXIT_NOT_REVIEWED
        if self.result.blocking:
            return EXIT_REQUEST_CHANGES
        return EXIT_NOT_REVIEWED if self.result.cut_short else EXIT_APPROVE  # cut short: never an approval

    def render(self) -> str:
        header = f"# {self.ref.owner}/{self.ref.repo}#{self.ref.number} at {self.head_sha[:7]}: {self.title}\n"
        if isinstance(self.result, CheckResult):
            return f"{header}\n**{self.result.title}**: {self.result.summary}\n"
        outcome = self.result
        verdict = "REQUEST CHANGES" if outcome.blocking else "PARTIALLY REVIEWED" if outcome.cut_short else "APPROVE"
        lines = [header, f"**Verdict: {verdict}** (nothing was posted to GitHub)\n"]
        for finding in outcome.posted:
            body = comment_body(finding, blocking=outcome.is_blocking(finding))
            lines.append(f"## {location(finding.path, finding.line)}\n\n{body}\n")
        if unchecked := outcome.unchecked:
            lines.append(f"**{len(unchecked)} candidate(s) were not checked before the time limit** (never posted):\n")
            lines.extend(f"- {location(f.path, f.line)}" for f in unchecked)  # where, not the unverified claim
            lines.append("")
        if failed := outcome.verifier_failed:
            lines.append(f"**{len(failed)} candidate(s) could not be verified** (the verifier failed):\n")
            lines.extend(f"- {location(f.path, f.line)} {f.title}: {reason}" for f, reason in failed)
            lines.append("")
        if outcome.low_confidence:
            lines.append(f"{len(outcome.low_confidence)} low-confidence candidate(s) were not verified:\n")
            lines.extend(
                f"- {location(f.path, f.line)} {f.title} (confidence {f.confidence})" for f in outcome.low_confidence
            )
            lines.append("")
        if outcome.rejected:
            lines.append(f"<details><summary>{len(outcome.rejected)} candidate(s) dropped by verification</summary>\n")
            lines.extend(f"- {location(f.path, f.line)} {f.title}: {reason}" for f, reason in outcome.rejected)
            lines.append("</details>")
        return "\n".join(lines) + "\n"


async def review_pr(
    ref: PullRef,
    *,
    reader: PullRequestReader,
    source: RepoSource,
    agents: AgentFactory,
    work_dir: Path,
    block_at: int = BLOCK_AT_PRIORITY,
    simplifier: SimplifierFactory | None = None,
    lenses: Sequence[Lens] = (),
    deadline: Deadline | None = None,
) -> Report:
    repo = Repo(installation_id=0, owner=ref.owner, name=ref.repo)  # no App installation involved
    state = await reader.get_pull(repo, ref.number)
    result = await review_change(
        reader=reader,
        source=source,
        agents=agents,
        work_dir=work_dir,
        repo=repo,
        number=ref.number,
        state=state,
        head_sha=state.head_sha,
        block_at=block_at,
        simplifier=simplifier,
        lenses=lenses,
        deadline=deadline,
    )  # no memory: on-demand reviews always start fresh
    return Report(ref, state.head_sha, state.title, result.outcome if isinstance(result, Reviewed) else result)


def run(argv: list[str] | None = None) -> int:
    from autoreviewer import logs, telemetry
    from autoreviewer.web.app import ModelPlan, build_lenses, correctness_agents, simplifier_agents
    from autoreviewer.web.config import Settings

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pr", help="https://github.com/owner/repo/pull/N or owner/repo#N")
    args = parser.parse_args(argv)  # bad usage exits 2, like every other error
    try:
        ref = parse_pr_ref(args.pr)
    except ValueError as error:
        parser.error(str(error))

    logs.configure()
    try:
        settings = Settings.from_env()
        telemetry.configure(service="autoreviewer-ondemand", include_content=settings.logfire_include_content)
        logs.configure(to_logfire=True)
        settings.work_dir.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("GITHUB_TOKEN") or None  # optional: private repos, higher API rate limit
        plan = ModelPlan.from_settings(settings)
        review = review_pr(
            ref,
            reader=PublicGitHub(token),
            source=PublicSource(token),
            agents=correctness_agents(plan),
            work_dir=settings.work_dir,
            block_at=settings.block_at_priority,
            simplifier=simplifier_agents(plan) if settings.simplify else None,
            lenses=build_lenses(settings, plan),
            deadline=Deadline(
                settings.soft_deadline_minutes * 60, checking_s=checking_cutoff(settings.hard_deadline_minutes * 60)
            ),
        )
        report = asyncio.run(asyncio.wait_for(review, settings.hard_deadline_minutes * 60))
    except Exception as error:  # any failure is an error, never a verdict
        logger.exception("on-demand review of %s failed", args.pr)
        print(f"error: could not review {args.pr}: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_ERROR
    print(report.render())
    return report.exit_code


def main() -> None:
    sys.exit(run())
