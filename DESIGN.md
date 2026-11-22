# Prism: design

A Macroscope-style GitHub PR reviewer, built on `pydantic-ai-harness`, served on
the homelab beside Boodleton and running on self-hosted models.

Status: **draft for Mike to review**. The open questions are in §12.

---

## 1. What it does (the Macroscope-ish contract)

| Behaviour | Rule |
|---|---|
| Trigger | PR `opened`, `reopened`, `ready_for_review`, `synchronize` (push), or a `@YOUR-APP-SLUG review` comment (e.g. PRs opened before the bot was installed; open PRs are never backfilled in bulk) |
| Lenses | **Correctness**, **DRY / don't-reinvent-the-wheel**, **Security** |
| Output | One GitHub review: a summary plus inline comments anchored to diff lines |
| Verdict | **Any open P0/P1 finding → `REQUEST_CHANGES`.** Otherwise `APPROVE` (P2/P3 findings are posted as non-blocking comments). `PRISM_BLOCK_AT_PRIORITY=3` makes every finding block |
| Noise policy | The lens is thorough and scores its own confidence; the verifier and code filter. Still **no nits**: style, naming, and taste are never findings |
| Conversation | Anyone (humans *and* bots) can reply in a finding's thread. The reviewer answers, holds its position, or withdraws the finding |
| Follow-up pushes | Re-checks open findings. Fixed ones get a "fixed in `abc123`" reply and the thread is resolved. Then it reviews only the new commits |
| Merge gate | Also publishes a **check run** (`autoreviewer`) that repos can mark as required |

> Why a check run as well as the review? Whether a GitHub App's
> `REQUEST_CHANGES` actually blocks a merge depends on each repo's
> branch-protection settings. A failing required check blocks it every time,
> so treat the check run as the gate and the review as the conversation.

### Not using the harness's `Macroscope` capability

`pydantic_ai_harness.Macroscope` shells out to the **hosted Macroscope CLI**:
their service and their models, on a Macroscope account. We want to *behave
like* Macroscope on *our* DeepSeek/Qwen, so that capability doesn't fit.

---

## 2. Architecture

```
GitHub ──webhook──▶ Cloudflare ──▶ Caddy :8443 ──▶ prism:8000  (FastAPI)
                                                     │  verify HMAC, route event
                                                     ▼
                                              DBOS durable queue (SQLite)
                                                     │  idempotent workflow ids
                     ┌───────────────────────────────┼─────────────────────────┐
                     ▼                               ▼                         ▼
              review_pr workflow           handle_reply workflow     installation workflow
                     │                               │
        fetch diff + tarball (installation token, held by CODE only)
                     │                               │
        ┌──── 3 lens agents (parallel) ────┐   conversation agent
        │ correctness │ DRY │ security     │   (thread history via StepPersistence)
        └──────────────┬───────────────────┘          │
                verifier agent (different model)      │
                       │                              │
                       ▼                              ▼
              Poster: pure code. Anchors comments, sets the verdict,
              writes the check run, resolves threads, updates the store
                                     │
                                     ▼
                 http://boodleton:8000/v1  ──▶ qwen3.8-flash-next / deepseek-v4.1-flash
```

**Main rule: the LLM never touches GitHub.** The agents get read-only file
tools on an extracted tarball and return **typed Pydantic output**.
Deterministic code turns that output into API calls. So:

- a prompt-injected PR can't make the bot merge, push, comment elsewhere, or read secrets;
- the verdict (`REQUEST_CHANGES` / `APPROVE`) is computed, not generated;
- the GitHub layer is unit-testable without a model (SRP).

That's why we deliberately **don't** use the harness `GitHub` (MCP)
capability. It would hand the model write tools carrying an installation token.

---

## 3. Harness building blocks (reuse > reinvent)

| Need | Use | Notes |
|---|---|---|
| Model access | `OpenAIChatModel` + `OpenAIProvider(base_url='http://boodleton:8000/v1')` | Internal container DNS on `homelab-public`, so traffic never goes back out through Cloudflare |
| Repo access for agents | `LocalWorkspace(clone_dir, read_only=True)` + `FileSystem(read_only=True, tools=[..., 'list_files', 'grep'])` | DRY lens *needs* grep to find existing helpers |
| Honour repo conventions | `RepoContext()` | Loads the target repo's `AGENTS.md`/`CLAUDE.md` so we review against *their* rules |
| Thread memory | `StepPersistence(store=SqliteStepStore(database=...))` with `conversation_id=<thread>` | Resume with `list_runs(conversation_id=…)` → `continue_run(...)` |
| Crash-safe jobs + queue | `DBOSDurability()` + DBOS queue (SQLite system DB) | Gives the async job queue, retries, and webhook-redelivery idempotency (`SetWorkflowID(delivery_id)`) without hand-rolling any of it |
| Per-tenant budgets | `SpendLimits(budgets=[Budget(tokens=…, window='day', scope=installation_id)])` | Our models cost $0/token, so budget **tokens**, not USD |
| Injection telemetry | `PromptInjectionDefender()` (report-only) | Flags suspicious PRs in logs. The real defence is the structure in §2 |
| GitHub API | **`githubkit`** (async, typed, does App JWT → installation tokens) | Don't hand-roll JWT signing or token caching |
| Evals | `pydantic_evals` | Precision/recall on a seeded-bug PR corpus, see §10 |

Not used, on purpose (YAGNI):
- `Shell`, sandboxes: the *review* path never executes PR code. Running code (repros, base-vs-head) is an opt-in stage with its own threat model: see [EXECUTION.md](EXECUTION.md).
- `Coder`: we never edit their code.
- `Subagents` / `DynamicWorkflow`: the pipeline is fixed, so orchestrate it in Python with `asyncio.gather`. Letting a model orchestrate a fixed pipeline just burns tokens and adds randomness.
- `Macroscope`, `GitHub` MCP: see above.

---

## 4. The review pipeline

1. **Gate.** Is the installation active and the repo enabled? Is the PR non-draft or the request explicit? Is this a new push or an explicit review request?
2. **Fetch** (code, installation token):
   - the diff: `pulls/{n}/files`, or `compare/{last_reviewed_sha}...{head}` for incremental runs (fall back to the full diff after a force-push);
   - the repo snapshot: `GET /repos/{o}/{r}/tarball/{head_sha}` into a size-capped tmpfs dir. No git binary, and no token in a remote URL.
3. **Filter.** Drop lockfiles, vendored, generated, and binary files. If the total changed lines exceed `MAX_DIFF_LINES` (≈3000), post one "split this PR" comment and set the check to `neutral`.
4. **Lenses** (in parallel). One factory, three instruction files. Each returns a report: every qualifying finding (not just the first), an overall verdict, and the files it inspected. A finding below confidence 4 is dropped without a verifier call.
5. **Verify.** A *different* model re-reads the cited code for each candidate and judges it against a fixed rubric (introduced by this PR, reachable trigger, no unstated assumptions, not intentional, actionable). It returns `confirmed | rejected` with a reason, and may correct the priority. Unconfirmed findings are dropped. Using a second model means the two models' mistakes are less likely to line up, and this step does most of the work on precision.
6. **Prove** (optional, see [EXECUTION.md](EXECUTION.md)). For confirmed high-severity findings, a repro test runs on head and on base in a throwaway sandbox. Findings that don't reproduce are dropped; bugs already present on base become non-blocking FYIs.
7. **Dedupe.** Each comment ends with a marker, `<!-- autoreviewer:<fingerprint> p<priority> <kind> -->`; the fingerprint is `sha256(head, kind, path, line)`. Before posting, `posting.Publisher` reads the bot's comments on the PR. A finding is skipped when a comment with its fingerprint is already there (a replayed review), when GitHub shows a comment of the same kind on the same path and line of this head (a superseded review's comment, carried over because the code there didn't change), or when the other lens already posted the same defect; in each case only if that comment is at least as urgent. A comment GitHub marks outdated covers nothing: the code changed.
8. **Anchor.** GitHub rejects review comments on lines outside the diff hunks (HTTP 422). Findings that can be anchored become inline comments. Findings on unchanged code (common for DRY: "this already exists in `utils/x.py`") are anchored to the *new* code that duplicates it. Anything that still can't be anchored goes into the review body.
9. **Stale check.** If the PR head moved while we were thinking, drop the result. The newer push has its own workflow.
10. **Post** (code): one review (`REQUEST_CHANGES` if any blocking finding is open, else `APPROVE`), the check run, and the store update.

### The lenses

| Lens | Looks for | Explicitly *not* |
|---|---|---|
| Correctness | logic errors, wrong edge cases, broken contracts with callers (grep the call sites!), race conditions, resource leaks, error handling that swallows failures | style, naming, "consider adding tests" |
| DRY | new code duplicating (a) code elsewhere in the PR, (b) an existing helper in the repo, (c) stdlib or an **already-declared** dependency | suggesting *new* dependencies; abstracting things that merely look alike (rule of three) |
| Security | injection (SQL, shell, path, template), authn/z gaps, SSRF, unsafe deserialisation, secrets in code, weak crypto, missing input validation at trust boundaries | theoretical issues with no reachable input |

Every lens must cite `path:line` evidence, and the verifier checks those
citations. Each finding carries a **priority** (P0-P3, from OpenAI Codex's review rubric) and a
calibrated **confidence** (0-10, from Qodo PR-Agent). Lessons taken from published reviewer prompts
(Codex, PR-Agent, Anthropic's security review, Gemini CLI):

- **Be thorough during review; filter afterwards.** The lens reports every finding with an honest
  confidence rather than suppressing doubtful ones, because a separate verifier and code-side
  thresholds do the filtering. Telling one model to both search and self-censor produced "approves
  everything".
- **A checklist and a method**, not just "find bugs": understand the change, read callers and the
  base version, then walk a list of bug classes for every changed line.
- **Explicit acceptance criteria** in the verifier, so precision is a rubric, not a mood.
- **Comments state the trigger first**, then the consequence. No praise, no "please verify".

### Security

`review_lens(kind="security")` in `review/pipeline.py`, run alongside correctness, always on the **whole PR**: a
continued review must re-check every vulnerability, or one found on an earlier commit would quietly stop
blocking. If it fails, the review is partial and never approves, since approving without it would be a
fail-open default.

- **Lens** (`prompts/security.md`, Qwen): learn the security model, compare the change against it, trace
  untrusted input to sensitive operations. Injection and code execution, authn/z, secrets and crypto,
  data exposure, insecure defaults. Excludes denial of service, resource leaks, rate limits, and
  test-only code. Method, categories, and exclusions are adapted from Anthropic's
  claude-code-security-review (MIT).
- **References by language**: `agents.security_references()` maps the PR's file extensions to Trail of
  Bits reference files, vendored unmodified in `prompts/vendor/trailofbits/` (CC-BY-SA-4.0): insecure
  defaults and auth pitfalls always, plus `sharp-edges/lang-<language>.md` per changed language. They are
  passed as run instructions: chosen by code, so the model can't skip them, and never read from the
  repository under review. (The skills' `SKILL.md` workflows need tools a read-only reviewer lacks, so
  only their reference files are used.)
- **Verifier** (`prompts/security_verifier.md`, DeepSeek): introduced by the change, attacker-reachable,
  concrete exploit with nothing on the path stopping it, not excluded. Adapted from Anthropic's findings
  filter.
- Findings share the P0-P3 gate. When both lenses report one line, the more urgent finding is posted.
  `PRISM_SECURITY=false` turns it off.

### DRY (advisory)

`prompts/dry.md`: new code that re-implements an existing helper in the repo, the standard library, a
dependency the project already declares, or other code in the same PR. Not new dependencies, not
look-alikes that behave differently, not a second copy with no helper yet (the rule of three), not test
files, and not unused imports or other leftovers (linters and the correctness lens cover those).
`prompts/dry_verifier.md` checks the existing thing is real and covers this use, edge cases included.
DRY findings never block (`ADVISORY_KINDS`), at most 5 (`MAX_DRY_POSTED`) are verified and posted per review,
the lens's best 5 by its own ranking (counting only findings past the free filters), and a continued review only looks at files changed since the last one.
`PRISM_DRY=false` turns it off.

**Docs and tests lenses** (Nov 18, off until their evals are in: `PRISM_DOCS`, `PRISM_TESTS`). Both
are advisory like DRY: never block, at most `MAX_ADVISORY_POSTED` (5) each, only files changed since the last
review when continuing. Both are found by DeepSeek and checked by Qwen, the other model. Kimi K3.1 Pro checked
them for a day (Nov 18-19) and was dropped when its hosted quota ran out twice.
- *docs*: text the change makes untrue: docstrings, comments, docs, config comments and documented settings,
  error and log messages, and contracts callers rely on (renamed flags, env vars, endpoints). The stale text
  is often in a file the PR didn't touch, so a finding cites the changed line that made it untrue and quotes
  the stale text with its location. It also holds the prose the PR writes to pstack's unslop rules
  (`vendor/pstack/skill-unslop.md`, verbatim, in both the lens's and the verifier's instructions): slop is P3,
  reported after drift, and the verifier confirms only a clear break of a cited rule with a rewrite that
  follows them all.
- *tests*: behaviour changes no test would catch breaking (the reviewer imagines the smallest break and reads
  the assertions), tests that can't fail, and weakened tests. It can't run tests; a sandbox could later make
  the mutations for real.

Evals: `--suite docs --lens docs` and `--suite tests --lens tests` (`evals/docs_cases.py`, `evals/tests_cases.py`):
5 seeded cases each on `d284fe8` (docs adds 2 slop cases), their own clean controls, and the correctness
suite's.

Security and DRY are both `Lens` records (`review/job.py`): a kind, an agent pair from
`agents.lens_pair()`, optional run instructions, and whether a continued review re-reads the whole PR.
Adding a lens is a prompt pair plus one line in `web/app.py: build_lenses`.

### Simplify (optional, one-click)

`review/simplify.py`, guided by pstack's principles (Lauren Tan's, MIT, vendored verbatim and pinned in
`prompts/vendor/pstack/`): laziness, subtract before you add, minimize reader load. Its list of comments
worth keeping is adapted from pstack's comment-sicko. It runs alongside the correctness lens.

1. Qwen proposes exact replacements `{path, start_line, end_line, replacement}` for **lines the PR adds**,
   aiming at roughly a fifth of them where the code allows, never inventing cuts to hit a number.
2. Code drops any it can prove wrong: outside the added lines, not shorter, overlapping, carrying a
   secret, or (Python) leaving the file unparseable (`compile(..., ONLY_AST)`, never executed).
3. DeepSeek checks each survivor behaves identically and is genuinely simpler.
4. Survivors are posted as GitHub multi-line `suggestion` comments (one click to apply). They **never
   block**, at any `PRISM_BLOCK_AT_PRIORITY`, and the review summary says how much of the added
   code they cut. A failure anywhere costs only the suggestions. `PRISM_SIMPLIFY=false` turns it off.

When a review continues on new commits, only files changed since the last review are considered, so
pushes don't repeat the same suggestions.

Measured with `python -m evals.simplify` (real functions from this repo, deliberately bloated; every
posted suggestion is applied and the full test suite run on the result). v1, 2026-11-17: **26 of 28
planted lines cut (93%)**, 4 of 4 bloated cases simplified, **0 test failures**, 0 suggestions on the
tight control case, about 13 GPU-minutes.

### Posting as findings are confirmed

Every lens has a `report_finding` tool (code-side only: the model still never touches GitHub). A
reported finding is verified at once, in parallel with the ongoing review, and posted as its own inline
comment the moment it's confirmed (`review/posting.py: Publisher`), so the author can start fixing while
the review continues. The final report only adds what wasn't reported along the way; type findings post
immediately, simplifications as each is checked.

The verdict still comes last: one review (REQUEST_CHANGES or APPROVE) whose inline comments are only the
findings that couldn't be posted early. Each early comment carries a hidden fingerprint
(`head|kind|path|line`), so a review replayed after a crash never posts one twice. When correctness and
security report one defect, a less urgent second report isn't posted; a more urgent one is. Findings
posted on a head that a newer push supersedes stay (GitHub marks them outdated), but that head gets no
verdict.

### Time limits and scope

Every lens gets one shared scope rule (`agents.FOCUS`): review the lines the PR adds or removes and what they
affect, and read the rest of the codebase only to understand them. Runaway reviews were scope drift, or a model stuck in a loop: verifiers get an 80-step budget
and every agent the harness's `DetectLoops`, so the deadlines are only a backstop.

`review/deadline.py`: the **soft deadline** defaults to 9 minutes (`PRISM_SOFT_DEADLINE_MINUTES`),
capped at 90% of the hard deadline. With five minutes left, lenses are asked to focus; with one minute
left, they lose their tools and must report what they found. Both windows shrink for shorter budgets. After the **hard deadline** (`PRISM_HARD_DEADLINE_MINUTES`, at most 10 per attempt) the job cancels the review:
"Review timed out", no verdict, and findings already posted stay. A review whose correctness or security lens was cut short, or failed outright (a model error after the
client's retries), **never approves**: it requests changes if it found blocking issues, and otherwise
posts a neutral COMMENT ("partially reviewed") with a neutral check, and isn't remembered for continuing
(files it never read wouldn't count as changed since). The CLI exits 3 for it, and the evals use the
bot's limits.

A running review checks every minute whether its PR moved on (a newer push, or closed) and stops at
once if so: its result would be thrown away, and the newer head's review may be waiting for its slot.
Crash recovery uses the original check and remaining time. A fresh explicit request after completion
starts a new check and budget, even if the commit already has a verdict. Publication recognizes the
check ID in the bot's review footer so recovery does not repeat that attempt's verdict.

Recovery also repeats no model work that finished (`review/checkpoints.py`, #60). Every agent run in an
attempt (each lens, checker, and the simplifier) checkpoints its settled turns, keyed by the DBOS workflow
id, role, candidate and model identity. After a restart a finished run returns its stored result and an
interrupted one continues from its last settled turn; a lens's reported findings are resubmitted so their
finished verdicts come back from their own checkpoints. Only the request in flight is sent again (with any
compaction that prepared it), so billing is not exactly-once. A run never resumes another model's or the
overflow namespace's checkpoint. The checkpoints are dropped once the attempt has its verdict, and with the
PR's memory when it closes.

### Deterministic checks (no model)

Rules a tool can decide exactly are a linter's job, not a lens's. `review/lint.py` runs ruff on the
Python files a PR changes and reports only on **lines the PR adds**: missing argument or return
annotations block (P1); `Any` and a bare `# type: ignore` are advisory (P2), because both are sometimes
justified. ruff only parses files and runs with `--isolated`, so the repo's own config (for example
`per-file-ignores`) can't switch the rules off. These findings skip the verifier, are labelled
"(types)", and aren't kept in the lens's memory, since they're recomputed every review. This repo applies
the same rules to itself (`pyproject.toml`). TypeScript (oxlint) is next once a TS repo is subscribed.

Unused imports (`F401`) are the other exception to "lines the PR adds": deleting an import's last use
leaves the import line untouched. So an unused import anywhere in a changed file is reported when the
same message isn't already raised at the merge-base, where ruff is run on `git show base:path`. New,
copied and renamed files have no base, so every unused import in them counts. These are P3 `lint`
findings and never block (`ADVISORY_KINDS`): failing the build is CI's job. The lens prompts tell
correctness and DRY to leave them alone; correctness once posted one as a blocking P1.
A type *checker* (`ty`) would need the PR's dependencies installed, which means running the PR's
build code, so it waits for the sandbox.

**A lens failing costs that lens, not the review** (Nov 19). #41's review died after eight minutes on one
request that Qwen refused, and every lens's work went with it. Each lens now runs under `job._survive`: a
failure is logged, the lens counts as having found nothing, and the others finish. An advisory lens costs only
its advice. A failed blocking lens makes the review partial (above), and the review says which one failed.

**Tool results are bounded where they're produced** (`compaction.tool_output_limits`). The compaction tiers
never shrink the newest results, so one over `COMPACT_AT / 8` (17.5k on Qwen) is spilled: the history keeps a preview and a
handle, and the reviewer pages through the rest with `read_tool_result`.

### Measuring it

`evals/` runs the real reviewer over seeded bugs in this repo's own code, with no GitHub calls:

    set -a; . ~/.config/prism/prism.env; set +a
    BOODLETON_BASE_URL=http://127.0.0.1:8484/v1 uv run python -m evals.run --label NAME

Isolated cases put one bug in a tiny PR. Buried cases plant one bug inside a real feature commit of
several hundred lines, which is what reviews actually face. Clean cases (refactors, and real commits
left untouched) measure noise. Any prompt or pipeline change should be measured before and after.

**Secret-handling rule:** if the security lens finds a committed secret, the
comment names the file and line but **never quotes the value**. The poster
also redacts anything that looks like a credential before posting.

---

## 5. Conversations (bots replying to the bot)

Events: `pull_request_review_comment` (thread replies) and `issue_comment`
(PR-level `@YOUR-APP-SLUG …`).

**Built so far: `@YOUR-APP-SLUG review`** (`web/routing.py` → `web/commands.py`).

- Accepted from the PR's author (human or bot) and from OWNER / MEMBER / COLLABORATOR; anyone else is
  ignored. The bot's own comments, edits, and quoted (`>`) lines never count as commands.
- The command targets the current head, including drafts. If an attempt is queued or running, the
  request joins it. After it finishes, a new command requests a fresh review of that head; redelivery
  of the same comment does not. A reaction acknowledges the command; reaction failures are cosmetic.
- A fresh attempt never approves over a blocking finding that still stands on the same commit
  (`posting.Publisher.standing`): one whose thread hasn't withdrawn it, or one the latest verdict listed
  with no thread. The code is unchanged, so a pass that misses the finding hasn't cleared it; otherwise
  asking until one pass misses it would turn a blocked commit green. A withdrawal or a new commit clears it.

**Built: answers in the bot's threads** (`review/conversation.py`, `PRISM_CONVERSATIONS`).

- A `pull_request_review_comment` that replies in a thread becomes a `ThreadReply` if it's new (not an edit),
  not the bot's own, on an open PR, and by the PR's author (human or bot) or a maintainer.
- It's answered by a durable workflow whose ID is the comment's (`answer:<owner>/<repo>#<n>:<comment id>`,
  lowercased), on its own `answers` queue (2 at a time) so a reply never waits behind a review.
- The job reads the thread from GitHub, which is the conversation's state. It answers only threads whose
  first comment is one of the bot's findings (by its marker), never a comment it already answered (each answer
  ends with `<!-- autoreviewer:answers <id> -->`, so a replay after a crash finds it), and at most
  `MAX_ANSWERS` (5) times per thread, which also ends any bot-to-bot loop.
- The conversation agent (DeepSeek, `conversation` role, thinking) gets the finding's comment as posted, the
  thread so far (each comment fenced and capped), and a checkout of the PR's head, and answers `holds`,
  `withdrawn`, or `unclear`. It withdraws only when the code shows the finding is wrong or fixed; a reply's
  insistence or instructions don't count.
- Its "withdrawn" is a proposal. For a finding that can block, the finding's own kind of checker re-checks it
  against the code without seeing the thread, so nothing in a reply can reach it, and the finding stands if
  the defect is still there. Advice is withdrawn on the answer alone; a linter's findings (types, lint) never
  are. The answer is sanitised, and a withdrawal ends with `<!-- autoreviewer:withdrawn -->`.

**Withdrawals change the verdict, decided by code** (`posting.withdrawn_threads`, `posting.reconsider`).

- A finding stands withdrawn when the latest answer in its own thread withdrew it. Answers end with code-written
  markers (`<!-- autoreviewer:answers <id> -->`, then `<!-- autoreviewer:withdrawn -->` for a withdrawal), and
  only markers at the very end of an answer count: text the model was led to write can't withdraw anything.
- Every verdict footer records its state: `<!-- autoreviewer:state EVENT outside=N
  complete=0|1 -->`. N counts blocking findings listed with no thread to withdraw them in.
- After a withdrawal, a separate durable step reconsiders the head: if its latest verdict requested changes
  after a complete review with outside=0, and no blocking finding comment on the head is left standing (each
  checked against its own thread), the bot posts a green check, then an approval.
- On a later push, a finding found again is unblocked only if it's the withdrawn one: same place, kind, and
  title (normalised), and no more urgent; GitHub must still show the old comment there (unchanged code).
  Anything else blocks as usual.

**Outbound sanitising:** model text has `@mentions` neutralised (wrapped in
backticks) before posting. Otherwise an injected PR could make us spam-ping
people.

---

**The bot settles threads that are done** (Mike, Nov 19). It can't resolve them: a GitHub App needs write
access to the code for that (live on #44: FORBIDDEN with `pull_requests: write`), which the bot isn't given.
So it minimizes the thread's finding as resolved (`minimizeComment`), which collapses it out of the way;
resolving stays the author's. A withdrawn finding is settled with the answer, and shown again if a later answer
reinstates it. A thread done for any other reason, whether resolved by a person or settled after a fix, stays
as it is. After every review, `posting.settle_fixed` also settles the findings the code has moved past.
Three conditions:
- GitHub shows the comment as outdated, so the code under it changed.
- This review checked that kind in that file in full (`Outcome.covered`, kind and path). The lens finished over
  that file, wasn't cut short and wasn't at its cap. Lint kinds count only when ruff ran under its cap and the
  correctness lens, which carries its findings, finished. A scoped lens covers only the files it re-read.
- It reported nothing of that kind in that file, in any state. That includes a finding folded into another
  lens's on the same line (`Outcome.merged`).

The rules are conservative on purpose: a reworded finding, or another one in the same file, keeps the thread
open. The finding is settled before the note goes up, so a failed update leaves no note claiming otherwise.
Nothing is settled on merge or close, so advice that was never acted on stays visible. Thread updates never
cost a review or an answer: a failure is logged.

## 6. State

Two stores, each with one job:

| Store | Holds | Why separate |
|---|---|---|
| `SqliteStepStore` (harness) | message history per thread conversation | Off the shelf, and resumable |
| `app.db` (ours, SQLite WAL) | domain truth: installs, PRs, findings | StepPersistence snapshots hold *messages only*, not domain state |

```sql
installations (id PK, account_login, account_type, status TEXT CHECK(status IN ('pending','active','suspended')),
               plan TEXT, created_at)
pull_requests (repo_id, number, last_reviewed_sha, check_run_id, verdict, updated_at,
               PRIMARY KEY (repo_id, number))
findings      (id PK, repo_id, pr_number, lens, severity, path, line, title, body, fingerprint UNIQUE,
               status TEXT CHECK(status IN ('open','fixed','withdrawn','dismissed')),
               comment_id, thread_node_id, created_sha, closed_sha, bot_turns INT DEFAULT 0)
```

Concurrency: a push and a reply on the same PR can race. Serialise per PR with
a short lease row, the same pattern as Boodleton's `work_timeline_store`
lease. Don't build a distributed lock for a single box.

---

## 7. Models & capacity

| Role | Default model | Why |
|---|---|---|
| Finders: `correctness`, `security`, `dry`, `simplify` | `qwen3.8-flash-next` | Local on the RTX PRO 6000, MTP drafting makes it fast |
| Checkers: `<kind>.verifier` | `deepseek-v4.1-flash` | Better, and a *different* model, so its mistakes are less likely to match the finder's |
| `docs`, `tests` and their checkers | DeepSeek finds, Qwen checks | Their evals ran this way; still two different models |

**Pinning.** Any role can be pinned to another model with one setting, e.g.
`PRISM_MODELS="security=deepseek-v4.1-flash,security.verifier=qwen3.8-flash-next"`; unlisted roles keep
the defaults (`config.models_from`). Each agent's context window follows its model (`servers.CONTEXT_WINDOWS`).
Startup stops on an unknown role, a model without a known window, or a finder pinned to the same model as its
checker.
`PRISM_REASONING="docs=high,tests=high"` sets a reasoning effort per role (low, medium, high).
DeepSeek doesn't think unless a request asks for an effort (any effort turns it on); Qwen thinks by
default. Unlisted roles send none. `web.app.ModelPlan` builds every agent's model, for the bot and the evals alike; the review footer names
each finder/checker pair.

**Capacity reality check:** Qwen serves about 3 concurrent 262k slots, and
**you share them** with your own Boodleton/Code Puppy use. One review is about
4+ agent runs with tool loops. So:
- set DBOS queue `worker_concurrency=2`;
- bound each run with a per-run `UsageLimits` (request + token caps);
- set per-installation daily token budgets (`SpendLimits`);
- expect a queue during busy hours, and say so on the landing page.

**Structured output on local models:** tool-call output parsing on
SGLang/TensorFold is the most likely flaky part. Use `output_retries=3`, keep
the output schemas flat, and record parse failures in the evals.

**Credential scoping gap:** Boodleton API keys have **no model allowlist**, so
the reviewer's key could reach the ChatGPT/Claude OAuth-backed models.
Reselling those subscriptions as a paid service would break their ToS. Either
add a `models` allowlist to Boodleton keys (small patch) or at minimum hard-pin
the model names in reviewer config. Prefer the allowlist.

---


**Lanes (measured 2026-11-18, widened Nov 20).** Qwen's server runs 5 streams of 168k tokens; DeepSeek's has
more but is shared. Every request goes through one process-wide limiter per model (`agents.with_lanes`, pydantic-ai's
`ConcurrencyLimitedModel`): `PRISM_MODEL_LANES`, default Qwen 5 and DeepSeek 4. Those defaults, and every other
model name and limit in this section, are the reference values in `deployment_profile.json`, which
`PRISM_DEPLOYMENT_PROFILE` replaces without a code change (docs/providers.md).

**Compaction points and reply room (Nov 20).** An agent compacts its history above `servers.COMPACT_AT`: 140k on
Qwen (Mike's call, higher than the old 60% of the window, so reviewers compact less), 60% of DeepSeek's 1M window.
Every request asks for at most `REPLY_TOKENS` (16,384): Qwen's server reserves `max_tokens` in the window before it
accepts a prompt, and 32,768 when a request doesn't say, which would cap prompts at 139k on a 168k window. Of 18,632
Qwen replies to the bot, the longest was 13,090 tokens. `COMPACT_AT + REPLY_TOKENS <= window` is checked when the
profile loads, and startup stops if it fails.

The Qwen server pre-budgets each stream's full window (fp8 KV, about 3.24 GiB at 168k; 4.32 GiB at 224k until
Nov 20) but only touches pages as contexts grow, so peak use stays well under the budget. Four full 262k windows don't fit, but four 224k windows do with the
engine's minimum 0.5 GiB reserve (`PARALLEL=4 CONTEXT=229376` until Nov 20); five 168k windows fit the same
17.6 GiB (`PARALLEL=5 CONTEXT=172032` in the recipe's `.env` since). In the day before the change,
the longest prompt was 216k (p99 136k, median 31k), so the cut rejected nothing. The thin reserve is the risk: if
all five streams fill at once the server could run out of memory. Rollback: delete the `.env` and restart
`qwen38-flash-next-tf`. Excess requests wait in
the bot rather than in the model server's queue, where everyone else waits too. One review runs at a time
(`PRISM_MAX_CONCURRENT_JOBS=1`): a review already runs three Qwen finders in parallel, and two at once left
6 of 9 requests queued at the server and pushed both past their deadline.

## 8. Subscriptions ("as a service")

A **GitHub App** install *is* the subscription unit. The user installs it on an
account or org and picks repos, and GitHub handles the consent UI.

**Today:** the App is private, so only its owner can install it, and the installation's repository
selection on GitHub is the *only* list of reviewed repos. (There used to be a second, bot-side
`PRISM_REPOS` allowlist; it drifted from the installation and silently ignored repos, so it was
removed.) **Before the App is made public, the `pending` approval gate below must exist**, or any
installer gets free GPU time.

- `installation` / `installation_repositories` webhooks → upsert `installations`.
- New installs start as `pending`, which stops randoms eating your GPU. You approve them, either from a tiny admin page or by plugging into **Boodleton's existing GitHub-OAuth access-request flow** (`access_control.py`). Reuse that rather than building a second user system.
- Uninstall → `deleted` event → purge that tenant's findings and histories.
- **Payments: decide later (YAGNI).** Options in §12. GitHub Marketplace paid plans have eligibility hurdles (verified publisher org, minimum installs), so it probably isn't day one.

App permissions (least privilege): **Pull requests: R/W, Checks: R/W,
Contents: R, Metadata: R.** Events: `pull_request`,
`pull_request_review_comment`, `pull_request_review`, `issue_comment`,
`check_run` (re-run button), `installation`, `installation_repositories`.

---

## 9. Homelab hosting (Boodleton-style)

- **Example hostname:** `review.example.com`, Cloudflare-proxied, routed to **Caddy** (8443 is already forwarded, so no new router hole):
  ```caddy
  review.example.com {
      encode zstd gzip
      reverse_proxy prism:8000
  }
  ```
- **Cloudflare gotcha:** the edge is already 403-ing `minecraft.` with some rule. Make sure WAF / Bot Fight Mode doesn't challenge GitHub's webhook POSTs; add a skip rule for `/webhook`. HMAC (`X-Hub-Signature-256`, constant-time compare) is the real authentication.
- **Live progress:** while a review runs, its check run (the "autoreviewer" row in the PR's Checks tab) shows the phase, model-request and tool-call counts, the latest thinking, and recent tool calls, updated at most every 30 s (`review/progress.py`, a capability watching each model response). The trail stays on the finished check run. Model text is sanitized and fenced in code blocks, so it can't ping anyone, leak a token, or inject markdown.
- **Telemetry:** Logfire (`telemetry.py`). Every pydantic-ai agent run (model requests, tool calls, tokens), every webhook request, and every log line, with the same token scrubbing as stdout. The receiver and on-demand CLI report as separate services. Off until `LOGFIRE_TOKEN` is in `prism.env`. Traces include the reviewed code by default; `PRISM_LOGFIRE_INCLUDE_CONTENT=false` keeps prompts and tool results out (needed before reviewing other people's private code).
- **Webhook handler:** verify HMAC, enqueue, return `202` well inside GitHub's 10-second timeout. Never do model work inline.
- **Quadlet** `deploy/web/prism.container`, copying Boodleton's hardening: `ReadOnly=true`, `DropCapability=all`, `NoNewPrivileges=true`, `Network=homelab-public.network`, tmpfs `/tmp` with a size cap (repo tarballs land here), `--memory=2g --pids-limit=256`, data volume `~/.local/share/prism:/var/lib/prism:Z`.
- **Secrets** in `~/.config/prism/` (files mode 0600): `prism-app.env` (GitHub App settings), `prism-app.pem` (private key, mounted read-only) and `prism.env` (provider credentials and receiver settings).
- Add the service to the README's routing table and update commands.

---

## 10. Repo layout

```
autoreviewer_bot/
  pyproject.toml                 # pydantic-ai-slim[openai,dbos], pydantic-ai-harness, githubkit, fastapi, uvicorn
  Containerfile
  src/autoreviewer/
    config.py                    # pydantic-settings: models, limits, secrets paths
    app.py                       # FastAPI: POST /webhook, GET /healthz, GET / (install landing)
    webhooks.py                  # HMAC verify + event → workflow routing (pure, unit-tested)
    github.py                    # githubkit wrapper: tokens, diff, tarball, review, check run, resolve thread
    workflows.py                 # DBOS workflows: review_pr, handle_reply, handle_installation
    pipeline.py                  # filter → lenses → verify → dedupe → anchor
    poster.py                    # Findings → GitHub calls; verdict; sanitising (no model imports!)
    diffmap.py                   # unified-diff parsing, line anchoring
    store.py                     # app.db access
    agents/
      schemas.py                 # Finding, Verification, ThreadReply
      factory.py                 # build_lens_agent(lens), build_verifier(), build_conversation_agent()
      prompts/{correctness,dry,security,verifier,conversation}.md
  tests/
  evals/                         # pydantic_evals datasets of seeded-bug PRs
```

The lenses are **data, not classes**. Adding a fourth lens means one prompt
file plus one enum value, with no new code paths (open/closed principle):

```python
class Lens(StrEnum):
    CORRECTNESS = 'correctness'
    DRY = 'dry'
    SECURITY = 'security'

def build_lens_agent(lens: Lens) -> Agent[ReviewDeps, list[Finding]]:
    return Agent(
        settings.lens_model,                      # OpenAIChatModel → boodleton
        deps_type=ReviewDeps,
        output_type=list[Finding],
        output_retries=3,
        instructions=(PROMPTS / f'{lens}.md').read_text(),
        capabilities=[
            FileSystem(read_only=True, tools=[*READ_TOOLS, 'list_files', 'grep']),
            RepoContext(),
            SpendLimits(budgets=settings.tenant_budgets),
            PromptInjectionDefender(),
        ],
    )
```

---

## 11. Milestones

| # | Ships | Done when |
|---|---|---|
| M0 | **Done 2026-11-16.** GitHub App, receiver at `/webhook` (signature check, repo allowlist, dedupe), quadlet, Caddy route | PR #18 in `pydantic-ai-claude-code` got the neutral "autoreviewer is listening" check run. Jobs run in an in-process `Dispatcher`; DBOS replaces it behind the same `submit()` in M1, when a review must survive a restart |
| M1 | **Done 2026-11-16.** Correctness lens (`qwen3.8-flash-next`) + verifier (`deepseek-v4.1-flash`) with read-only file tools over a safe checkout, computed verdict, inline comments, check run, stale-head check; **DBOS** workflow per PR head (`review:owner/repo#N@sha`, SQLite, queue concurrency 2, max 3 recoveries) | PR #19: seeded bug flagged, fix approved. PR #20: container SIGKILLed mid-review, DBOS recovered it with one check run and one review |
| **Eval gate** | `pydantic_evals` corpus of ~30 PRs (seeded bugs, duplicated helpers, injected vulns, plus clean PRs) | Precision ≥ 0.8 on the clean set, *before* adding lenses |
| M2 | DRY + security lenses, fingerprints, incremental re-review, auto-resolve fixed threads | The eval corpus passes per lens |
| M3 | Thread conversations, bot-turn caps, human dismiss | A bot author can argue a DRY finding away; ping-pong is capped |
| M4 | Installation approval flow, tenant budgets, landing page | A stranger can install it and land in `pending` |
| M5 | Running code: repros, base-vs-head, test-verified `fixed` | See [EXECUTION.md](EXECUTION.md) section 9 |

The eval gate matters. A reviewer that blocks merges with false positives
gets uninstalled in a week, and that's exactly the pitfall Macroscope
competes on.

---

## 12. Open questions for Mike

1. **Who can subscribe?** Allowlist-approved (pending → active), or open to anyone?
2. **Money?** Free for approved users / Stripe later / GitHub Marketplace later?
3. **Clean PR verdict:** `APPROVE`, or just a passing check plus a `COMMENT` summary?
4. **Model split:** is Qwen for the lenses and DeepSeek for verifying/chatting OK, given the GPU sharing?
5. **App slug:** choose a unique name during GitHub App registration.
6. **Boodleton key allowlist patch:** OK to add a per-key `models` allowlist to Boodleton (§7)?
