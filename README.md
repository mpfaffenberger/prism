<p align="center"><img src="docs/brand/prism.webp" width="192" alt="Prism: a single beam splits through an emerald prism"></p>
<h1 align="center">Prism</h1>
<p align="center"><strong>One model finds the bug. Another checks the claim.</strong><br>A GitHub pull request reviewer you can talk back to.</p>

<p align="center">
  <a href="pyproject.toml"><img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-76E6BD?labelColor=112D32"></a>
  <a href="src/autoreviewer/review/agents.py"><img alt="Built on pydantic-ai-harness" src="https://img.shields.io/badge/built_on-pydantic--ai--harness-76E6BD?labelColor=112D32"></a>
  <a href="docs/providers.md"><img alt="Bring your own models" src="https://img.shields.io/badge/models-bring_your_own-DDEA76?labelColor=112D32"></a>
</p>

## On a pull request

Each review attempt gets at most **10 minutes of review time** from the start of its GitHub check. Waiting in the queue does not count; restarting the receiver does not reset the clock, and the review picks up at its last completed model turn instead of starting over. Slow GitHub delivery or retrying the final check update can delay when the verdict appears. If a blocking review runs out of time it never approves.

Prism reads the diff **and the code around it**. When the finder flags a correctness bug, a separate checker checks whether the claim holds up in the repository before it becomes an inline finding. The models don't hold GitHub credentials; application code posts the review and sets the verdict.

| What it looks for | The question behind the comment |
| --- | --- |
| Correctness | Which input makes the new behavior wrong? |
| Security | Can data or an action cross a boundary it shouldn't? |
| DRY | Is there already a helper here that handles this exact case? |
| Docs & tests (opt-in) | Which documented promise changed? Would a test notice? |

**A suggestion isn't a veto.** High-priority correctness and security findings can request changes, as can blocking typing-lint findings. Reuse suggestions stay advisory. An incomplete blocking review doesn't earn an approval. To enforce the verdict at merge time, make the `autoreviewer` check required in your repository rules.

## Talk back, in the thread

Already have the App installed? Open a PR, push a commit, or comment `@YOUR-APP-SLUG review` on an existing PR. If a review of the current commit is queued or running, the command joins it. Once it finishes, a new command starts a fresh check with its own time budget, even on the same commit. A fresh attempt cannot approve over a blocking finding that still stands on that commit: contest it in its thread, or push a fix. Redelivering the original request does not start another attempt.

Reply **on the finding**. A PR-opening bot can do this just like a person can.

> **Example author reply:** “This helper only gets normalized paths. Can you check the caller in `resolve_path`?”

Prism keeps the conversation with the finding. It can explain why the issue still holds or withdraw it when the code supports the objection. A withdrawn comment can be collapsed without marking the GitHub thread resolved.

## Inside Prism

[![Architecture diagram showing signed GitHub events, durable review jobs, finder and checker models, private state, thread replies, the preview CLI, and the isolated sandbox daemon.](docs/brand/architecture.svg)](docs/brand/architecture.svg)

The [review agents](src/autoreviewer/review/agents.py) use Pydantic AI models. [Durable workflows](src/autoreviewer/review/durable.py) track review work across webhook deliveries.

## Make your own reviewer

**Bring your own models.** Any [Pydantic AI model](https://ai.pydantic.dev/models/) that supports tool calls and structured output can be a Prism reviewer. Codex and Claude are examples, not a whitelist.

From a checkout with [uv](https://docs.astral.sh/uv/) installed, run `uv run prism-config`. Enter two tool-capable models you have access to: a finder and an independent checker. Or script it:

```sh
uv run prism-config --finder openai-codex:gpt-6.2-sol --checker claude-code:claude-opus-5-5 --output prism.json
```

**Already serving models locally?** vLLM, llama.cpp (`llama-server`), SGLang, TensorFold, and other OpenAI-compatible Chat Completions servers use the same setup. Replace the two IDs with distinct models from your server's `/v1/models` list:

```sh
uv run prism-config --local-url http://localhost:8000/v1 \
  --finder local:FINDER_MODEL_ID --checker local:CHECKER_MODEL_ID
```

Pass `--local-context N` to use your server’s configured token window instead of the conservative 4096-token compaction trigger. Use an address reachable **from the receiver**. For separate servers, container networking, or an authenticated endpoint, see [Local inference](docs/providers.md#local-inference).

Point the receiver's `PRISM_CONFIG` at that file. Keep provider credentials in the receiver environment, not in Git or the JSON. See the [provider guide](docs/providers.md) for Codex, Copilot, Claude and custom models. Then [register and install your GitHub App](docs/github-app.md): choose your own name and hostname, connect the webhook, and try a review. No shared inference server is required.

To work on Prism locally, install [uv](https://docs.astral.sh/uv/) and run these from the checkout:

```sh
uv sync --locked
uv run pytest
```

The default test suite skips tests that boot real microVMs. For model-backed quality measurements, see the [eval runner and case suites](evals/).

[Architecture & design notes](DESIGN.md) · [GitHub App setup](docs/github-app.md) · [Web service deployment](deploy/web/)
