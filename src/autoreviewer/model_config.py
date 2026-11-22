"""Portable model selection. Only load operator-owned configuration: factories execute Python."""

from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def _local_model(name: str, base_url: str) -> Model:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    # A nonempty placeholder keeps the SDK from falling back to OPENAI_API_KEY.
    key = os.environ.get("PRISM_LOCAL_API_KEY", "").strip() or "prism-local"
    return OpenAIChatModel(name, provider=OpenAIProvider(base_url=base_url, api_key=key))


def _claude_code_model(name: str) -> Model:
    url = os.environ.get("PRISM_CLAUDE_CODE_BASE_URL", "").strip()
    key = os.environ.get("PRISM_CLAUDE_CODE_API_KEY", "").strip()
    if url or key:
        # Never send a gateway key to a default URL or inherit an unrelated OpenAI credential.
        if not url or not key:
            raise ValueError("claude-code requires PRISM_CLAUDE_CODE_BASE_URL and PRISM_CLAUDE_CODE_API_KEY")
        from autoreviewer.review.agents import boodleton_model

        return boodleton_model(name, base_url=url, api_key=key)

    try:
        from pydantic_ai_claude_code import ClaudeCodeModel
    except ModuleNotFoundError as exc:
        if exc.name != "pydantic_ai_claude_code":
            raise
        raise ValueError(
            "Claude Code needs the claude-code extra: uv sync --extra claude-code (Python 3.13-3.14)"
        ) from exc
    return ClaudeCodeModel(name)


def check_model_spec(spec: str) -> str:
    """`spec` if it is `provider:model` or `factory:module:function`; the one definition of a model spec."""
    provider, sep, name = spec.partition(":")
    if not sep or not provider or not name or any(c.isspace() for c in spec):
        raise ValueError("expected provider:model or factory:module:function")
    if provider == "factory":
        module, _, function = name.partition(":")
        if not function.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("expected factory:module:function")
    return spec


def preferred_roles(pool: dict[str, int], finder: str | None, checker: str | None) -> tuple[str | None, str | None]:
    if finder is None:
        finder = next((spec for spec in pool if spec != checker), None)
    if checker is None:
        checker = next((spec for spec in pool if spec != finder), None)
    return finder, checker


class PrismConfig(BaseModel):
    """Per-model options are keyed by the full model spec, not the role."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    finder: str
    checker: str
    overflow: str | None = None
    pool: Annotated[dict[str, Annotated[int, Field(gt=0)]], Field(min_length=2)] | None = None
    checks: Literal["pool", "checker"] = "pool"
    """Where a pool places checks: on the pool, or on the dedicated `checker` (the pool only places finders then,
    and checks of a finder on the checker's own model fall back to the pool)."""
    local_context: dict[str, Annotated[int, Field(ge=16)]] = Field(default_factory=dict)
    endpoints: dict[str, str] = Field(default_factory=dict)
    lanes: dict[str, Annotated[int, Field(ge=1)]] = Field(default_factory=dict)
    compact_at: dict[str, Annotated[int, Field(ge=8)]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def default_roles(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("checks") == "checker" and not data.get("checker"):
            raise ValueError('checks="checker" needs an explicit checker')
        if isinstance(data, dict) and isinstance(pool := data.get("pool"), dict):
            finder, checker = preferred_roles(pool, data.get("finder"), data.get("checker"))
            return {"finder": finder, "checker": checker, **data}
        return data

    @field_validator("finder", "checker", "overflow")
    @classmethod
    def model_spec(cls, spec: str | None) -> str | None:
        return None if spec is None else check_model_spec(spec)

    @field_validator("endpoints")
    @classmethod
    def base_urls(cls, endpoints: dict[str, str]) -> dict[str, str]:
        for spec, url in endpoints.items():
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or not parsed.path.strip("/")
                or parsed.port == 0  # accessing port also rejects malformed or out-of-range ports
                or parsed.username is not None
                or any(c in url for c in "?#\\")
                or any(c.isspace() or not c.isprintable() for c in url)
            ):
                raise ValueError(
                    f"endpoint for {spec} must be an HTTP(S) base URL with an API prefix "
                    "and without credentials, query or fragment"
                )
        return endpoints

    @model_validator(mode="after")
    def check_models(self) -> Self:
        if self.finder == self.checker:
            raise ValueError("finder and checker must use different models")
        pool = self.pool or {}
        if self.checks == "checker" and not pool:
            raise ValueError('checks="checker" only applies to a pool')
        for spec, capacity in pool.items():
            check_model_spec(spec)
            if spec in self.lanes and self.lanes[spec] != capacity:
                raise ValueError(f"lanes for {spec} conflict with pool capacity {capacity}")
        specs = {self.finder, self.checker, *pool}
        if self.overflow is not None:
            if self.overflow == self.checker and self.checks == "pool":  # with a dedicated checker, the pool checks it
                raise ValueError("overflow finder and checker must use different models")
            specs.add(self.overflow)
        local_specs = {spec for spec in specs if spec.startswith("local:")}
        if self.local_context.keys() - local_specs:
            raise ValueError("local_context only applies to configured local models")
        if missing := local_specs - self.endpoints.keys():
            raise ValueError(f"missing endpoints for local models: {sorted(missing)}")
        if extra := self.endpoints.keys() - local_specs:
            raise ValueError(f"endpoints only apply to configured local models: {sorted(extra)}")
        unknown = (self.lanes.keys() | self.compact_at.keys()) - specs
        if unknown:
            raise ValueError(f"options for unconfigured models: {sorted(unknown)}")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        return cls.model_validate_json(Path(path).read_bytes())

    def lane_count(self, spec: str) -> int:
        if self.pool is not None and spec in self.pool:
            return self.pool[spec]
        return self.lanes.get(spec, 1)

    def resolve(self, spec: str) -> tuple[Model, int]:
        # A model that only takes overflow exists to absorb a queue, so it is uncapped unless lanes are set.
        # One that is also the finder or a pool member keeps that cap: lanes belong to the model, not the role.
        overflow_only = spec == self.overflow != self.finder and spec not in (self.pool or {})
        return resolve_model(
            spec,
            endpoint=self.endpoints.get(spec),
            local_context=self.local_context.get(spec),
            compact_at=self.compact_at.get(spec),
            lanes=self.lanes.get(spec) if overflow_only else self.lane_count(spec),
        )


def resolve_model(
    spec: str,
    *,
    endpoint: str | None = None,
    local_context: int | None = None,
    compact_at: int | None = None,
    lanes: int | None = None,
) -> tuple[Model, int]:
    from pydantic_ai.models import Model, infer_model

    from autoreviewer.review.agents import with_lanes

    if spec.startswith("factory:"):
        _, module, function = spec.split(":")
        model = getattr(import_module(module), function)()
        if not isinstance(model, Model):
            raise TypeError(f"{spec} must return a pydantic_ai Model")
    elif spec.startswith("local:"):
        if endpoint is None:
            raise ValueError(f"missing endpoint for {spec}")
        model = _local_model(spec.removeprefix("local:"), endpoint)
    elif spec.startswith("claude-code:"):
        model = _claude_code_model(spec.split(":", 1)[1])
    else:
        model = infer_model(spec)

    # Leave at least half of a known window for output, tool schemas and counting error.
    # For unknown models use a small default; the operator can supply a measured target.
    window = local_context if spec.startswith("local:") else model.profile.get("context_window")
    limit = None if window is None else min(window // 2, window - ((model.settings or {}).get("max_tokens") or 0))
    target = compact_at if compact_at is not None else limit if limit is not None else 4096
    if target < 8 or (limit is not None and target > limit):
        raise ValueError(f"compact_at={target} for {spec} leaves insufficient room in its {window}-token window")
    if lanes is not None:
        model = with_lanes(model, spec, lanes)
    return model, target
