"""Receiver settings, read once from the environment (the quadlet's EnvironmentFiles)."""

from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from autoreviewer.model_config import PrismConfig
from autoreviewer.review.deadline import DEFAULT_SOFT_S, MAX_REVIEW_S
from autoreviewer.roles import AGENT_KINDS, AGENT_ROLES, verifier_role
from autoreviewer.roles import CONVERSATION as CONVERSATION
from autoreviewer.roles import AgentKind as AgentKind
from autoreviewer.servers import (
    CONTEXT_WINDOWS,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_LANES,
    DEFAULT_MODELS,
    DEFAULT_OVERFLOW_MODEL,
    DEFAULT_REASONING,
    REASONING_EFFORTS,
    ReasoningEffort,
)

DEFAULT_PRIVATE_KEY_PATH = "/run/prism/github-app.pem"


@dataclass(frozen=True)
class Settings:
    """Which repos the bot reviews is decided in one place only: the App installation's repository
    selection on GitHub. The App is private, so only its owner can install it."""

    app_id: int
    app_slug: str
    webhook_secret: str = field(repr=False)
    private_key: str = field(repr=False)
    logfire_include_content: bool = True
    """Send prompts, model output, and tool results (the reviewed code) to Logfire with each trace."""
    security: bool = True
    """Run the security lens alongside correctness (see review/pipeline.py, review_lens)."""
    dry: bool = True
    """Run the DRY lens: new code that re-implements an existing helper, the stdlib, or a dependency."""
    docs: bool = False
    """Run the docs lens: docstrings, comments, docs, config comments, error messages, and contracts the change
    makes untrue. Off until its evals are in."""
    conversations: bool = False
    """Answer replies under the bot's findings (see review/conversation.py)."""
    tests: bool = False
    """Run the tests lens: changed behaviour no test would catch breaking, and tests that can't fail. Off until
    its evals are in."""
    simplify: bool = True
    """Post optional one-click simplifications of the code a PR adds (see review/simplify.py)."""
    block_at_priority: int = 1
    """Confirmed findings at this priority or more urgent (0 = P0 ... 3 = P3) block the merge; 3 blocks on all."""
    soft_deadline_minutes: float = DEFAULT_SOFT_S / 60
    """After this, every lens wraps up with what it has; a review cut short never approves."""
    hard_deadline_minutes: float = MAX_REVIEW_S / 60
    """The attempt's whole budget. Checking stops shortly before it (deadline.checking_cutoff) so the verdict is
    posted from what was judged; at the limit itself whatever still runs is cancelled, without a verdict."""
    model_lanes: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_MODEL_LANES))
    """At most this many requests at once to each model, across every review (PRISM_MODEL_LANES,
    "model=n,..."; defaults in servers.DEFAULT_MODEL_LANES). Models not listed are unlimited."""
    max_concurrent_jobs: int = 1
    """One review at a time: each already runs a finder per lens in parallel, about as many as the finders' model
    has lanes (servers.DEFAULT_MODEL_LANES). Two at once left most requests waiting and pushed both past their
    deadline."""
    boodleton_base_url: str = DEFAULT_BASE_URL
    boodleton_api_key: str = field(default="", repr=False)
    models: Mapping[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    """The model each agent role runs on (see models_from); its context window follows (servers.CONTEXT_WINDOWS)."""
    reasoning: Mapping[str, ReasoningEffort] = field(default_factory=lambda: dict(DEFAULT_REASONING))
    """The reasoning effort sent for each role that has one (see reasoning_from); unlisted roles send none."""
    work_dir: Path = Path("/var/lib/prism/work")
    database: Path = Path("/var/lib/prism/dbos.sqlite")
    """DBOS system database: queued and in-flight reviews survive restarts here."""
    steps_dir: Path = Path("/var/lib/prism/steps")
    """Per-PR lens conversations, so a new commit continues the last review (see review/memory.py)."""

    prism: PrismConfig | None = None
    overflow_repos: frozenset[str] = frozenset()
    """Repositories whose reviews may run on the overflow model when the normal workers are busy
    (see overflow_repos_from); `{"*"}` is every repository."""
    overflow_model: str | None = None
    """The model spec whose finders take those reviews: prism.json's `overflow` under PRISM_CONFIG, else the
    deployment profile's `overflow_model`. Never the profile's under PRISM_CONFIG, so a portable deployment
    cannot send code to a provider its own config didn't name."""

    @property
    def bot_login(self) -> str:
        return f"{self.app_slug}[bot]"

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> Settings:
        key_path = Path(env.get("PRISM_PRIVATE_KEY_PATH", DEFAULT_PRIVATE_KEY_PATH))
        prism_path = env.get("PRISM_CONFIG", "").strip()
        prism = PrismConfig.from_file(prism_path) if prism_path else None
        return cls(
            prism=prism,
            overflow_repos=overflow_repos_from(env),
            overflow_model=prism.overflow if prism else DEFAULT_OVERFLOW_MODEL,
            app_id=int(env["GITHUB_APP_ID"]),
            app_slug=env["GITHUB_APP_SLUG"],
            webhook_secret=env["GITHUB_WEBHOOK_SECRET"],
            private_key=key_path.read_text(),
            max_concurrent_jobs=int(env.get("PRISM_MAX_CONCURRENT_JOBS", cls.max_concurrent_jobs)),
            model_lanes=(
                {spec: prism.lane_count(spec) for spec in (prism.finder, prism.checker)}
                if prism
                else model_lanes_from(env)
            ),
            soft_deadline_minutes=(deadlines := deadlines_from(env))[0],
            hard_deadline_minutes=deadlines[1],
            block_at_priority=int(env.get("PRISM_BLOCK_AT_PRIORITY", "1")),
            simplify=parse_bool(env, "PRISM_SIMPLIFY", default=True),
            security=parse_bool(env, "PRISM_SECURITY", default=True),
            dry=parse_bool(env, "PRISM_DRY", default=True),
            docs=parse_bool(env, "PRISM_DOCS", default=False),
            tests=parse_bool(env, "PRISM_TESTS", default=False),
            conversations=parse_bool(env, "PRISM_CONVERSATIONS", default=False),
            logfire_include_content=parse_bool(env, "PRISM_LOGFIRE_INCLUDE_CONTENT", default=True),
            boodleton_base_url=env.get("BOODLETON_BASE_URL", cls.boodleton_base_url),
            boodleton_api_key="" if prism else env["BOODLETON_API_KEY"],
            models=(
                {role: prism.finder if role in AGENT_KINDS else prism.checker for role in AGENT_ROLES}
                if prism
                else models_from(env)
            ),
            reasoning=dict(DEFAULT_REASONING) if prism else reasoning_from(env),
            work_dir=Path(env.get("PRISM_WORK_DIR", str(cls.work_dir))),
            database=Path(env.get("PRISM_DATABASE", str(cls.database))),
            steps_dir=Path(env.get("PRISM_STEPS_DIR", str(cls.steps_dir))),
        )


OVERFLOW_REPOS = "PRISM_OVERFLOW_REPOS"
LEGACY_OVERFLOW_REPOS = "PRISM_SOL_OVERFLOW_REPOS"
"""OVERFLOW_REPOS's name from when one overflow model was built in. Still read, so a deployed unit keeps working."""


def overflow_repos_from(env: Mapping[str, str]) -> frozenset[str]:
    """PRISM_OVERFLOW_REPOS: comma-separated exact owner/repo names, or `*` for every repository. Blank is off.
    Which model takes the overflow is configured separately (prism.json's `overflow`, or the deployment profile)."""
    value, legacy = (env.get(key, "").strip() for key in (OVERFLOW_REPOS, LEGACY_OVERFLOW_REPOS))
    if value and legacy and value != legacy:
        raise ValueError(f"{OVERFLOW_REPOS} and {LEGACY_OVERFLOW_REPOS} disagree; set only {OVERFLOW_REPOS}")
    value = value or legacy
    if not value:
        return frozenset()
    repos = set[str]()
    if value == "*":
        return frozenset({"*"})
    for entry in value.split(","):
        repo = entry.strip()
        if not re.fullmatch(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*/[a-zA-Z0-9._-]+", repo) or repo.endswith(("/.", "/..")):
            raise ValueError(f"{OVERFLOW_REPOS}: expected exact owner/repo, got {entry!r}")
        repos.add(repo.lower())
    return frozenset(repos)


_TRUE, _FALSE = frozenset({"true", "1", "yes", "on"}), frozenset({"false", "0", "no", "off"})


def parse_bool(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    """Strict: a typo must stop startup, not silently pick a side (e.g. keep sending code to Logfire)."""
    value = env.get(name, "").strip().lower()
    if not value:
        return default
    if value in _TRUE | _FALSE:
        return value in _TRUE
    raise ValueError(f"{name}={env[name]!r}: expected true/false (or 1/0, yes/no, on/off)")


def model_lanes_from(env: Mapping[str, str]) -> dict[str, int]:
    """PRISM_MODEL_LANES, or the defaults when it's missing or blank (a placeholder line must never switch
    the limits off). Shared by the bot and the evals."""
    return parse_lanes(env.get("PRISM_MODEL_LANES", "")) or dict(DEFAULT_MODEL_LANES)


def parse_lanes(value: str) -> dict[str, int]:
    """ "model=3,other=4" -> {"model": 3, "other": 4}; a malformed entry stops startup. Blank -> {}."""
    lanes: dict[str, int] = {}
    for name, count in _pairs(value, "PRISM_MODEL_LANES", "model=lanes (lanes >= 1)"):
        if not count.isdigit() or int(count) < 1:
            raise ValueError(f"PRISM_MODEL_LANES: expected model=lanes (lanes >= 1), got {name}={count}")
        lanes[name] = int(count)
    return lanes


def models_from(env: Mapping[str, str]) -> dict[str, str]:
    """Each agent role's model: DEFAULT_MODELS, with PRISM_MODELS ("role=model,...") pinning roles. Reject an unknown
    role, a model without a known context window, or a finder on the same model as its checker (different models
    make different mistakes; that's the point of checking). Shared by the bot and the evals."""
    models = dict(DEFAULT_MODELS)
    for role, model in _role_pairs(env, "PRISM_MODELS", "role=model"):
        if model not in CONTEXT_WINDOWS:
            raise ValueError(f"PRISM_MODELS: no known context window for {model!r} (servers.CONTEXT_WINDOWS)")
        models[role] = model
    for kind in AGENT_KINDS:
        if models[kind] == models[verifier_role(kind)]:
            raise ValueError(f"PRISM_MODELS: {kind} and {verifier_role(kind)} are both {models[kind]!r}")
    return models


def deadlines_from(env: Mapping[str, str]) -> tuple[float, float]:
    """Return (soft, hard) minutes for the receiver and evals. Both settings must be positive and finite.

    Cap the hard limit at `deadline.MAX_REVIEW_S`, and the soft limit at the same share of the effective hard
    limit that the defaults use, so checking what the lenses found always keeps its share of the time.
    """
    soft = float(env.get("PRISM_SOFT_DEADLINE_MINUTES", Settings.soft_deadline_minutes))
    hard = float(env.get("PRISM_HARD_DEADLINE_MINUTES", Settings.hard_deadline_minutes))
    if not all(math.isfinite(value) and value > 0 for value in (soft, hard)):
        raise ValueError("review deadlines must be positive finite minutes")
    hard = min(hard, MAX_REVIEW_S / 60)
    return min(soft, hard * DEFAULT_SOFT_S / MAX_REVIEW_S), hard


def reasoning_from(env: Mapping[str, str]) -> dict[str, ReasoningEffort]:
    """PRISM_REASONING ("role=effort,...", e.g. "docs=high,tests=high"): the reasoning effort each role's
    requests carry, over DEFAULT_REASONING. Other roles send none, which leaves the server's default: DeepSeek
    doesn't think then (any effort turns it on), while Qwen thinks by default. Stops startup on an unknown role or
    effort."""
    # str -> its ReasoningEffort literal: a lookup the type checker can follow, where `in` wouldn't narrow.
    efforts: dict[str, ReasoningEffort] = {effort: effort for effort in REASONING_EFFORTS}
    reasoning: dict[str, ReasoningEffort] = dict(DEFAULT_REASONING)
    for role, value in _role_pairs(env, "PRISM_REASONING", "role=effort"):
        if (effort := efforts.get(value)) is None:
            raise ValueError(f"PRISM_REASONING: {role}={value!r}; efforts: {', '.join(REASONING_EFFORTS)}")
        reasoning[role] = effort
    return reasoning


def _role_pairs(env: Mapping[str, str], setting: str, expected: str) -> list[tuple[str, str]]:
    """`setting`'s "role=value" pairs; an unknown role stops startup."""
    pairs = _pairs(env.get(setting, ""), setting, expected)
    for role, _ in pairs:
        if role not in DEFAULT_MODELS:
            raise ValueError(f"{setting}: unknown role {role!r}; roles: {', '.join(AGENT_ROLES)}")
    return pairs


def _pairs(value: str, setting: str, expected: str) -> list[tuple[str, str]]:
    """ "a=1, b=2," -> [("a", "1"), ("b", "2")]; blank entries are skipped, a malformed one stops startup."""
    pairs = []
    for item in filter(None, (part.strip() for part in value.split(","))):
        key, _, val = item.rpartition("=")
        if not key.strip() or not val.strip():
            raise ValueError(f"{setting}: expected {expected}, got {item!r}")
        pairs.append((key.strip(), val.strip()))
    return pairs
