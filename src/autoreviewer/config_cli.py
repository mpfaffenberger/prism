"""Write an operator-owned Prism configuration without loading a provider or handling secrets."""

import argparse
from pathlib import Path

from pydantic import ValidationError

from autoreviewer.model_config import PrismConfig, preferred_roles


def _pool_model(value: str) -> tuple[str, int]:
    spec, sep, count = value.rpartition("=")
    if not sep or not spec or not count.isascii() or not count.isdecimal():
        raise argparse.ArgumentTypeError("expected SPEC=CAPACITY with a positive integer capacity")
    return spec, int(count)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finder", help="provider:model that finds problems")
    parser.add_argument("--checker", help="different provider:model that verifies findings")
    parser.add_argument("--overflow", help="optional overflow finder spec, independent of the checker")
    parser.add_argument("--overflow-url", metavar="URL", help="override the endpoint for a local:<id> overflow finder")
    parser.add_argument("--overflow-lanes", type=int, help="maximum concurrent overflow requests (default: uncapped)")
    parser.add_argument(
        "--pool-model",
        type=_pool_model,
        action="append",
        metavar="SPEC=CAPACITY",
        help="add a pool model, most preferred first (repeat for each model)",
    )
    parser.add_argument(
        "--checks",
        choices=("pool", "checker"),
        help="with --pool-model: place checks on the pool (default) or always on --checker",
    )
    parser.add_argument("--local-context", type=int, metavar="TOKENS", help="served window per local model")
    parser.add_argument("--local-url", metavar="URL", help="default endpoint for local:<id> models")
    parser.add_argument("--finder-url", metavar="URL", help="override the endpoint for a local:<id> finder")
    parser.add_argument("--checker-url", metavar="URL", help="override the endpoint for a local:<id> checker")
    parser.add_argument("--finder-lanes", type=int, help="maximum concurrent finder requests (default: 1)")
    parser.add_argument("--checker-lanes", type=int, help="maximum concurrent checker requests (default: 1)")
    parser.add_argument("--output", type=Path, default=Path("prism.json"), help="new file (default: prism.json)")
    args = parser.parse_args(argv)
    try:
        pool: dict[str, int] = dict(args.pool_model or [])
        if args.checks is not None and not pool:
            parser.error("--checks requires --pool-model")
        if args.checks == "checker" and args.checker is None:
            parser.error("--checks checker requires --checker")
        if args.pool_model:
            if len(pool) != len(args.pool_model):
                parser.error("duplicate --pool-model spec")
            if len(pool) < 2:
                parser.error("--pool-model requires at least two distinct specs")
            finder, checker = preferred_roles(pool, args.finder, args.checker)
            assert finder is not None and checker is not None  # at least two distinct pool members
        else:
            finder = args.finder or input("Finder model (provider:model): ").strip()
            checker = args.checker or input("Checker model (provider:model): ").strip()
        roles = [
            ("finder", finder, args.finder_url, args.finder_lanes),
            ("checker", checker, args.checker_url, args.checker_lanes),
            ("overflow", args.overflow, args.overflow_url, args.overflow_lanes),
        ]
        lanes = {}
        endpoints = {}
        for role, spec, url, count in roles:
            if spec is None:
                if url is not None or count is not None:
                    parser.error(f"--{role}-url and --{role}-lanes require --{role}")
                continue
            if count is not None and lanes.setdefault(spec, count) != count:
                parser.error(f"conflicting lane counts for {spec}")
            if not spec.startswith("local:"):
                if url is not None:
                    parser.error(f"--{role}-url requires --{role} to use local:<id>")
                continue
            if url is not None and endpoints.setdefault(spec, url) != url:
                parser.error(f"conflicting endpoints for {spec}")
        if args.local_url is not None:
            # The shared default fills every local model (role or pool member) that has no URL of its own.
            local = {spec for spec in (finder, checker, args.overflow, *pool) if spec and spec.startswith("local:")}
            if not (remaining := local - endpoints.keys()):
                parser.error("--local-url needs a local:<id> model without a role-specific URL")
            endpoints.update(dict.fromkeys(remaining, args.local_url))
        options: dict[str, object] = {"finder": finder, "checker": checker, "overflow": args.overflow, "lanes": lanes}
        if pool:
            options["pool"] = pool
        if args.checks is not None:
            options["checks"] = args.checks
        if args.local_context is not None:
            if not endpoints:
                parser.error("--local-context requires a local model endpoint")
            options["local_context"] = dict.fromkeys(endpoints, args.local_context)
        if endpoints:
            options["endpoints"] = endpoints
        config = PrismConfig.model_validate(options)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(config.model_dump_json(indent=2, exclude_defaults=True) + "\n")
    except (ValidationError, OSError, EOFError) as exc:
        parser.exit(2, f"Configuration not completed: {exc}\n")
    print(f"Wrote {args.output.resolve()}. Set PRISM_CONFIG to its path inside the receiver.")
