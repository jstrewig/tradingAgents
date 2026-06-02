# Known Issues and Deferred Work

These items were identified during the mid-tier LLM feature review and deferred for future attention. None block current functionality; they are recorded here to prevent rediscovery.

---

## Q1 - Env-skip is all-or-nothing across thinking tiers

**Location:** `cli/main.py` — Step 7 (Thinking Agents) block around line 608.

**Behavior:** The interactive thinking-agent prompt is skipped when _any_ of `TRADINGAGENTS_QUICK_THINK_LLM`, `TRADINGAGENTS_MID_THINK_LLM`, or `TRADINGAGENTS_DEEP_THINK_LLM` is set in the environment. Tiers whose env var is _not_ set fall back to their `DEFAULT_CONFIG` values rather than prompting. A user who sets only `TRADINGAGENTS_DEEP_THINK_LLM` intending to pin just the deep tier will find quick and mid silently defaulted as well.

**Current mitigation:** The console confirmation line already prints all three resolved values:

```
✓ Thinking agents from environment: quick=<model>, mid=<model>, deep=<model>
```

**Proposed fix (deferred):** Per-tier prompt logic — skip only the tiers whose env var is set and prompt for the rest. This is a meaningful UX change; the current or-gate is preserved for backward compatibility with the two-variable gate it replaced.

---

## Q7 - Provider kwargs (reasoning_effort, thinking_level, effort) are tier-unaware

**Location:** `tradingagents/graph/trading_graph.py` — `_get_provider_kwargs` (line 146) and LLM construction (lines 88-109).

**Behavior:** `_get_provider_kwargs` returns a single `llm_kwargs` dict that is spread into all three LLM client constructors (quick, mid, deep). When `openai_reasoning_effort` is set, it is forwarded to the mid-tier client even though the default mid model (`gpt-4.1`) is a non-reasoning model that does not use the parameter. OpenAI currently ignores unknown kwargs on non-reasoning models, so this produces no error in practice, but it is a latent mismatch.

**Proposed fix (deferred):** Build per-tier kwargs dicts and apply reasoning/thinking config only to tiers whose assigned model is a reasoning model. This requires understanding which models are reasoning-capable, which is a non-trivial classification problem across providers.

**Trigger to prioritize:** If a provider begins returning an error for unexpected kwargs on non-reasoning model calls.

---

## Q8 - GraphSetup takes mid_thinking_llm as a positional argument

**Location:** `tradingagents/graph/setup.py` — `GraphSetup.__init__` signature.

**Behavior:** `mid_thinking_llm` was inserted as the 2nd positional argument (between `quick_thinking_llm` and `deep_thinking_llm`). Today the only call site is `trading_graph.py`, so nothing is broken. However, this is a positional-signature change to a non-private class; any external code constructing `GraphSetup` positionally would break silently.

**Proposed fix (deferred):** Make `mid_thinking_llm` keyword-only (e.g., `*, mid_thinking_llm`), or append it after `deep_thinking_llm` to minimise positional disruption. Safe to defer as long as `GraphSetup` has no external callers.

---

## Q9 - No graph-level test asserts mid-tier routing in setup.py

**Location:** `tradingagents/graph/setup.py` — `create_agents` and risk-debator construction.

**Behavior:** There is no unit test asserting that `GraphSetup` wires `create_fundamentals_analyst` and the aggressive/neutral/conservative risk debators to `mid_thinking_llm` rather than `quick_thinking_llm` or `deep_thinking_llm`. The existing config and CLI tests cover the config key and env-var override, but not the factory wiring.

**Why deferred:** `GraphSetup` builds real LLM clients at construction time, making it awkward to unit-test the wiring without either mocking the client factory or incurring API calls. The config-level and CLI-level tests are judged sufficient for regression protection on this change.

**Proposed fix (deferred):** Patch `create_llm_client` (or introduce a dependency-injection seam) so `GraphSetup` can be constructed in tests without live API calls, then assert that each agent factory receives the expected LLM instance.

---

## Q10 - pytest is not declared as a dev dependency

**Location:** `pyproject.toml` (or equivalent) — no `[project.optional-dependencies]` or `requirements-dev.txt` exists.

**Behavior:** `pytest` is required to run the test suite but is not declared anywhere. A fresh checkout has no automated signal that `pytest` must be installed; `pip install .` does not pull it in.

**Proposed fix (deferred, propose separately):** Add a dev extras group to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8"]
```

Or add a `requirements-dev.txt` with `pytest>=8`. This is independent of the mid-tier feature and should be proposed as a standalone change.
