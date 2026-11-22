"""HTTP edge: verify the signature, drop duplicates, route, answer fast. Everything slow is a job."""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from githubkit.webhooks import verify

from autoreviewer.model_config import PrismConfig, resolve_model
from autoreviewer.model_pool import Choice, ModelPool
from autoreviewer.servers import COMPACT_AT, ReasoningEffort
from autoreviewer.web.config import AGENT_KINDS, CONVERSATION, OVERFLOW_REPOS, AgentKind, Settings, verifier_role

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

    from autoreviewer.github import GitHubApp
    from autoreviewer.review.conversation import ConversationDeps
    from autoreviewer.review.job import AgentFactory, Lens, ReviewDeps
    from autoreviewer.review.schemas import LensKind, LensReport, SimplifyVerdict, Verification
    from autoreviewer.review.simplify import SimplifierFactory
    from autoreviewer.review.workspace import Workspace
from autoreviewer.web.routing import (
    Ignore,
    MalformedPayload,
    Pong,
    PullRequestClosed,
    PullRequestUpdated,
    ReviewRequested,
    ThreadReply,
    route,
)

logger = logging.getLogger("autoreviewer.web")

PullRequestHandler = Callable[[PullRequestUpdated], Awaitable[object]]
ClosedHandler = Callable[[PullRequestClosed], Awaitable[None]]
RequestHandler = Callable[[ReviewRequested], Awaitable[object]]
ReplyHandler = Callable[[ThreadReply], Awaitable[object]]


def _reply(status_code: int, **body: str) -> JSONResponse:
    return JSONResponse(body, status_code=status_code)


def create_app(
    settings: Settings,
    on_pull_request: PullRequestHandler,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
    on_pull_request_closed: ClosedHandler | None = None,
    on_review_requested: RequestHandler | None = None,
    on_thread_reply: ReplyHandler | None = None,
) -> FastAPI:
    """`on_pull_request` must be quick (it enqueues); `lifespan` starts and stops whatever runs the work."""
    app = FastAPI(title="autoreviewer", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhook")
    async def webhook(request: Request) -> JSONResponse:
        body = await request.body()
        signature = request.headers.get("x-hub-signature-256", "")
        # sha256 only: githubkit's verify would accept a legacy sha1 signature, which we never want.
        if not signature.startswith("sha256=") or not verify(settings.webhook_secret, body, signature):
            logger.warning("rejected a delivery with a missing or wrong signature")
            return _reply(401, error="invalid signature")
        event = request.headers.get("x-github-event")
        delivery = request.headers.get("x-github-delivery")
        if not event or not delivery:
            return _reply(400, error="missing X-GitHub-Event or X-GitHub-Delivery")
        try:
            decision = route(event, json.loads(body), settings.app_slug)
        except (ValueError, MalformedPayload) as error:
            logger.warning("delivery %s: malformed %s payload: %s", delivery, event, error)
            return _reply(400, error="malformed payload")

        logger.info("delivery %s: %s -> %s", delivery, event, decision)
        match decision:
            case Pong():
                return _reply(200, status="pong")
            case Ignore(reason=reason):
                return _reply(200, status="ignored", reason=reason)
            case PullRequestUpdated() as pull:
                # Awaited, not fire-and-forget: if enqueueing fails, GitHub sees a 500 and can redeliver.
                # Duplicate deliveries are harmless; the review workflow's ID makes them no-ops.
                await on_pull_request(pull)
                return _reply(202, status="accepted")
            case PullRequestClosed() as closed:
                if on_pull_request_closed is not None:
                    await on_pull_request_closed(closed)
                return _reply(200, status="closed")
            case ReviewRequested() as command:
                if on_review_requested is None:
                    return _reply(200, status="ignored", reason="commands are not enabled")
                await on_review_requested(command)  # a failure is a 500, so GitHub's redelivery can retry it
                return _reply(202, status="accepted")
            case ThreadReply() as reply:
                if on_thread_reply is None:
                    return _reply(200, status="ignored", reason="conversations are not enabled")
                await on_thread_reply(reply)  # queues the answer; its workflow ID makes redeliveries no-ops
                return _reply(202, status="accepted")

    return app


@dataclass(frozen=True)
class ModelPlan:
    """The models, compaction targets and request limits used to build the review agents."""

    models: Mapping[str, str]
    base_url: str
    api_key: str = field(repr=False)
    lanes: Mapping[str, int]
    reasoning: Mapping[str, ReasoningEffort] = field(default_factory=dict[str, "ReasoningEffort"])
    """Role -> reasoning effort (config.reasoning_from)."""

    prism: PrismConfig | None = None
    pool: ModelPool | None = None
    overflow_finder: tuple[Model, int] | None = None
    """Pins every finder (with a pool, too); checkers stay the plan's own."""
    _resolved: dict[str, tuple[Model, int]] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelPlan:
        return cls(
            settings.models,
            settings.boodleton_base_url,
            settings.boodleton_api_key,
            settings.model_lanes,
            settings.reasoning,
            settings.prism,
            ModelPool(settings.prism) if settings.prism and settings.prism.pool else None,
        )

    def finder(self, kind: AgentKind) -> tuple[Model, int]:
        """The model for `kind`'s finder (a lens, or the simplifier's proposer), and where it compacts."""
        return self._build(kind)

    def verifier(self, kind: AgentKind) -> tuple[Model, int]:
        """The model for the agent that checks `kind`'s findings, and where it compacts."""
        return self._build(verifier_role(kind))

    def conversation(self) -> tuple[Model, int]:
        """The model for answering replies under the bot's findings, and where it compacts."""
        return self._build(CONVERSATION)

    def _build(self, role: str) -> tuple[Model, int]:
        """Resolve the role through the configured provider or the legacy Boodleton plan."""
        from autoreviewer.review.agents import boodleton_model

        name = self.models[role]
        if self.overflow_finder is not None and role in AGENT_KINDS:
            return self.overflow_finder
        if self.prism is not None:
            if name not in self._resolved:
                self._resolved[name] = self.prism.resolve(name)
            return self._resolved[name]
        if role not in self._resolved:
            self._resolved[role] = (
                boodleton_model(
                    name,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    lanes=self.lanes.get(name),
                    reasoning=self.reasoning.get(role),
                ),
                COMPACT_AT[name],
            )
        return self._resolved[role]


def build_agent_factory(settings: Settings) -> AgentFactory:
    """(lens, verifier) for correctness over a checkout, on the configured models: shared by the bot and the CLI."""
    return correctness_agents(ModelPlan.from_settings(settings))


def correctness_agents(plan: ModelPlan) -> AgentFactory:
    from autoreviewer.review.agents import correctness_agent, verifier_agent

    if plan.pool:
        return pooled_lens(plan.pool, "correctness", plan.overflow_finder)
    lens_model, lens_compact_at = plan.finder("correctness")
    verifier_model, verifier_compact_at = plan.verifier("correctness")
    return lambda workspace: (
        correctness_agent(lens_model, workspace, compact_at=lens_compact_at),
        verifier_agent(verifier_model, workspace, compact_at=verifier_compact_at),
    )


MAX_ADVISORY_POSTED = 5
"""Each advisory lens (DRY, docs, tests) posts at most its 5 best findings. The eval's clean 786-line commit drew 8
DRY comments, mostly true but too many to read; 3 dropped real findings for little gain; 5 keeps most of them
while a PR's blocking findings stay easy to see (Mike, Nov 18)."""

LENS_NAMES: dict[LensKind, str] = {"security": "security", "dry": "DRY", "docs": "docs", "tests": "tests"}
"""How the review footer names each lens; lenses run and are listed in this order."""


def enabled_lenses(settings: Settings) -> list[LensKind]:
    """The lenses switched on beside correctness, in LENS_NAMES order: Settings has one switch per lens, named after
    it."""
    return [kind for kind in LENS_NAMES if getattr(settings, kind)]


def build_lenses(settings: Settings, plan: ModelPlan | None = None) -> tuple[Lens, ...]:
    """The enabled lenses beside correctness, each on its own models (Settings.models)."""
    plan = plan or ModelPlan.from_settings(settings)
    return tuple(lens(kind, lens_agents(plan, kind)) for kind in enabled_lenses(settings))


def lens_agents(plan: ModelPlan, kind: LensKind) -> AgentFactory:
    """(lens, verifier) for `kind` over a checkout, on the plan's models for it: shared by the bot and the evals."""
    from autoreviewer.review.agents import lens_pair

    if plan.pool:
        return pooled_lens(plan.pool, kind, plan.overflow_finder)
    lens_model, lens_compact_at = plan.finder(kind)
    verifier_model, verifier_compact_at = plan.verifier(kind)
    return lambda workspace: lens_pair(
        kind,
        lens_model,
        verifier_model,
        workspace,
        lens_compact_at=lens_compact_at,
        verifier_compact_at=verifier_compact_at,
    )


def pooled_lens(pool: ModelPool, kind: Literal["correctness"] | LensKind, pinned: Choice | None = None) -> AgentFactory:
    from autoreviewer.review.agents import checker_agent, correctness_agent, extra_finder

    def agents(workspace: Workspace) -> tuple[Agent[None, LensReport], Callable[[], Agent[None, Verification]]]:
        model, target = pinned or pool.choose()
        finder = (
            correctness_agent(model, workspace, compact_at=target)
            if kind == "correctness"
            else extra_finder(kind, model, workspace, compact_at=target)
        )

        def check() -> Agent[None, Verification]:
            verifier, compact = pool.check(model.model_name)
            return checker_agent(kind, verifier, workspace, compact_at=compact)

        return finder, check

    return agents


def select_correctness(pool: ModelPool) -> tuple[AgentFactory, str]:
    from autoreviewer.review.agents import lens_version

    choice = pool.choose()
    return pooled_lens(pool, "correctness", choice), lens_version("pool:" + pool.key(choice[0]))


def lens(kind: LensKind, agents: AgentFactory) -> Lens:
    """A lens with its production settings, for the bot and the evals alike. Security reviews the whole PR every
    time (it can block); the others are advice: the changed files only when continuing, and at most
    MAX_ADVISORY_POSTED."""
    from autoreviewer.review.agents import security_references
    from autoreviewer.review.job import Lens

    if kind == "security":
        return Lens("security", agents, instructions=security_references)
    return Lens(kind, agents, whole_pr=False, max_posted=MAX_ADVISORY_POSTED)


def build_simplifier_factory(settings: Settings) -> SimplifierFactory | None:
    """None when simplifications are off."""
    return simplifier_agents(ModelPlan.from_settings(settings)) if settings.simplify else None


def simplifier_agents(plan: ModelPlan) -> SimplifierFactory:
    """Proposals and checks on the plan's models for the simplifier."""
    from autoreviewer.review.agents import simplify_agent, simplify_checker_agent
    from autoreviewer.review.simplify import Simplifier

    if (pool := plan.pool) is not None:

        def pooled(workspace: Workspace) -> Simplifier:
            model, target = plan.overflow_finder or pool.choose()

            def check() -> Agent[None, SimplifyVerdict]:
                verifier, compact = pool.check(model.model_name)
                return simplify_checker_agent(verifier, workspace, compact_at=compact)

            return Simplifier(simplify_agent(model, workspace, compact_at=target), check)

        return pooled
    propose_model, propose_compact_at = plan.finder("simplify")
    check_model, check_compact_at = plan.verifier("simplify")
    return lambda workspace: Simplifier(
        simplify_agent(propose_model, workspace, compact_at=propose_compact_at),
        simplify_checker_agent(check_model, workspace, compact_at=check_compact_at),
    )


def models_note(settings: Settings) -> str:
    """The review footer's "correctness and security by `a`, verified by `b`", one clause per pair of models."""
    if settings.prism and settings.prism.pool:
        return _pool_note(settings.prism)
    enabled: list[tuple[AgentKind, str]] = [
        ("correctness", "correctness"),
        *((kind, LENS_NAMES[kind]) for kind in enabled_lenses(settings)),
    ]
    groups: dict[tuple[str, str], list[str]] = {}
    for kind, name in enabled:
        groups.setdefault((settings.models[kind], settings.models[verifier_role(kind)]), []).append(name)
    return "; ".join(
        f"{_and(names)} by `{finder}`, verified by `{checker}`" for (finder, checker), names in groups.items()
    )


def _pool_note(config: PrismConfig, finder: str | None = None) -> str:
    """How a pool review's models are named in its footer; `finder` is an overflow finder pinned beside the pool."""
    pool = ", ".join(config.pool or {})
    checker = config.checker if config.checks == "checker" and finder != config.checker else None
    if finder is None:
        return (
            f"found by model pool: {pool}; verified by `{checker}`" if checker else f"scheduled from model pool: {pool}"
        )
    return f"found by `{finder}`, " + (
        f"verified by `{checker}`" if checker else f"checks scheduled from model pool: {pool}"
    )


def _and(names: list[str]) -> str:
    return ", ".join(names[:-1]) + f" and {names[-1]}" if len(names) > 1 else names[0]


def thread_reply_handler(settings: Settings) -> ReplyHandler | None:
    """Replies under the bot's findings are queued for an answer only with PRISM_CONVERSATIONS on."""
    from autoreviewer.review import durable

    return durable.enqueue_answer if settings.conversations else None


def conversation_deps(settings: Settings, github: GitHubApp, plan: ModelPlan | None = None) -> ConversationDeps:
    """Answers on the conversation role's model, over a fresh checkout of the PR's head."""
    from autoreviewer.review.agents import checker_agent, conversation_agent
    from autoreviewer.review.conversation import ConversationDeps
    from autoreviewer.review.workspace import GitHubSource

    plan = plan or ModelPlan.from_settings(settings)
    model, compact_at = plan.conversation()

    def checker(workspace: Workspace, kind: str) -> Agent[None, Verification]:
        """The finding's own kind of checker; only kinds that can block are re-checked (security, correctness)."""
        checked: Literal["correctness", "security"] = "security" if kind == "security" else "correctness"
        checker_model, checker_compact_at = plan.verifier(checked)
        return checker_agent(checked, checker_model, workspace, compact_at=checker_compact_at)

    return ConversationDeps(
        github=github,
        source=GitHubSource(github),
        agents=lambda workspace: conversation_agent(
            model, workspace, compact_at=compact_at, thinking=plan.reasoning.get(CONVERSATION) if plan.prism else None
        ),
        checkers=checker,
        work_dir=settings.work_dir,
        bot_login=settings.bot_login,
        block_at=settings.block_at_priority,
    )


def overflow_deps(settings: Settings, normal: ReviewDeps, plan: ModelPlan) -> ReviewDeps | None:
    """Change only finders; keep the normal checkout policy, checkers and review limits."""
    from autoreviewer.review.agents import lens_version

    if not settings.overflow_repos:
        return None
    spec = settings.overflow_model
    if spec is None:
        raise ValueError(
            f"{OVERFLOW_REPOS} is set but no overflow model is configured: "
            "add `overflow` to prism.json, or `overflow_model` to the deployment profile"
        )
    # A pool places every check on a model other than its finder's (ModelPool.check).
    if plan.pool is None and any(plan.models[verifier_role(kind)] == spec for kind in AGENT_KINDS):
        raise ValueError(f"{OVERFLOW_REPOS}: overflow checker cannot also be {spec}")
    # A factory or another provider spec can expose the same underlying model.
    active: list[AgentKind] = ["correctness", *enabled_lenses(settings)]
    if settings.simplify:
        active.append("simplify")
    finder = plan.prism.resolve(spec) if plan.prism is not None else resolve_model(spec)
    for kind in active if plan.pool is None else ():
        if plan.verifier(kind)[0].model_name == finder[0].model_name:
            raise ValueError(f"{OVERFLOW_REPOS}: {kind} checker resolves to the overflow finder")
    models = {**plan.models, **dict.fromkeys(AGENT_KINDS, spec)}
    # Shares the normal plan's resolved checker clients (or its pool, which then places only checks).
    overflow = replace(plan, models=models, overflow_finder=finder)
    return replace(
        normal,
        agents=correctness_agents(overflow),
        select_correctness=None,  # The overflow finder is pinned; the pool must not pick another.
        simplifier=simplifier_agents(overflow) if settings.simplify else None,
        lenses=build_lenses(settings, overflow),
        models_note=(
            _pool_note(settings.prism, spec)
            if settings.prism and settings.prism.pool
            else models_note(replace(settings, models=models))
        ),
        session_namespace="overflow",  # Its own sessions: no local transcript is sent to an overflow finder.
        lens_version=lens_version(spec),
    )


def main() -> None:
    import logfire
    import uvicorn

    from autoreviewer import logs, telemetry
    from autoreviewer.github import GitHubApp, Repo
    from autoreviewer.review import durable
    from autoreviewer.review.agents import lens_version
    from autoreviewer.review.job import ReviewDeps
    from autoreviewer.review.memory import ReviewMemory, conversation_id
    from autoreviewer.review.workspace import GitHubSource
    from autoreviewer.web.commands import request_review

    settings = Settings.from_env()
    telemetry.configure(service="autoreviewer", include_content=settings.logfire_include_content)
    logs.configure(to_logfire=True)
    settings.work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("starting as %s; reviewing the repositories selected on the App installation", settings.bot_login)

    github = GitHubApp.from_key(settings.app_id, settings.private_key)
    memory = ReviewMemory(settings.steps_dir)
    plan = ModelPlan.from_settings(settings)
    deps = ReviewDeps(
        github=github,
        source=GitHubSource(github),
        agents=correctness_agents(plan),
        work_dir=settings.work_dir,
        bot_login=settings.bot_login,
        models_note=models_note(settings),
        memory=memory,
        select_correctness=(lambda: select_correctness(pool)) if (pool := plan.pool) else None,
        lens_version=lens_version(settings.models["correctness"], settings.reasoning.get("correctness")),
        block_at=settings.block_at_priority,
        simplifier=simplifier_agents(plan) if settings.simplify else None,
        lenses=build_lenses(settings, plan),
        soft_deadline_s=settings.soft_deadline_minutes * 60,
        hard_deadline_s=settings.hard_deadline_minutes * 60,
    )
    overflow = overflow_deps(settings, deps, plan)
    durable.configure(settings.database)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Also recovers interrupted reviews and answers.
        await durable.launch(
            deps,
            conversation_deps(settings, github, plan),
            concurrency=settings.max_concurrent_jobs,
            overflow_deps=overflow,
            overflow_repos=settings.overflow_repos,
        )
        try:
            yield
        finally:
            durable.shutdown()

    async def review_requested(request: ReviewRequested) -> None:
        # An explicit request retries a review of this commit that was cancelled or crashed.
        await request_review(github, request, durable.enqueue_review)

    async def forget(closed: PullRequestClosed) -> None:
        repo = Repo(closed.installation_id, closed.owner, closed.repo)
        memory.forget(conversation_id(repo, closed.number))

    app = create_app(
        settings,
        durable.enqueue_review,
        lifespan,
        forget,
        review_requested,
        thread_reply_handler(settings),
    )
    logfire.instrument_fastapi(app)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")  # only reachable on homelab-public
