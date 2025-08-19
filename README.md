# LLM Market Sim

An LLM-driven multi-agent financial market simulator with a limit order book (LOB), orchestration,
and an evaluation suite. Designed to be *runnable out-of-the-box* (no API keys required) with a
fallback "heuristic LLM" for decisions, while optionally supporting OpenAI / Gemini.

---

## Features (MVP)
- Event-driven LOB (price-time priority); limit & market orders, partial fills, cancels.
- Agents:
  - `LLMDecisionAgent` (low-frequency intent → high-frequency execution)
  - Baselines: `MarketMakerAgent`, `MomentumAgent`, `NoiseAgent`
- Execution policies: simple TWAP/Immediate mapper from intent to orders.
- Synthetic price/flow generators (OU + regime shifts + jump toggles).
- Evaluation: PnL, inventory, spread/depth, returns stats (kurtosis, ACF), market quality sketches.
- CLI one-liner to run scenarios and write results (Parquet/CSV) + quick plots.

## Quickstart

```bash
# 1) create venv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) install
pip install -e ".[dev]"

# 3) run a small scenario (no API keys required)
python -m scripts.run_experiment --scenario small --steps 2000 --seed 42

# 4) see outputs
ls outputs/
#  - events.parquet   # order/trade timeline
#  - accounts.parquet # agent PnL/inventory over time
#  - snapshots.parquet# book snapshots (coarse)
#  - plots/           # quick diagnostic PNGs
````

## Optional: real LLMs

Set environment variables to enable real LLM calls for the LLM agent:

```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini  # or any chat model id

# Google Gemini
export GEMINI_API_KEY=...
export GEMINI_MODEL=gemini-1.5-flash

# DeepSeek
export DEEPSEEK_API_KEY=...
export DEEPSEEK_MODEL=deepseek-chat

# ChatGLM (ZhipuAI)
export CHATGLM_API_KEY=...
export CHATGLM_MODEL=glm-4-flash
```

If no keys are found, the LLM agent falls back to a deterministic **heuristic** model so the
simulation remains reproducible and cheap.

## Repo Layout

```
llm-market-sim/
  exchange/            # LOB & models
  agents/              # LLM + baselines + execution
  orchestration/       # simulator/event loop
  signals/             # synthetic generators
  eval/                # metrics & plots
  benchmarks/          # scenario factories
  scripts/             # entrypoints/CLI
  outputs/             # (created at runtime)
```

## Paper-style checklist

* **Stylized facts**: heavy tails, volatility clustering → see `eval/metrics.py`.
* **Market quality**: spreads, depth, slippage → `eval/metrics.py`.
* **Ablations**: toggle agent mixes via `--scenario` and `--mix-json`.
* **Scaling**: increase `--agents` + `--steps` to observe stability/throughput.
* **Reproducibility**: `--seed` controls generators & agent sampling.

## CLI

```bash
python -m scripts.run_experiment \
  --scenario medium \
  --steps 5000 \
  --seed 123 \
  --llm-interval 20 \
  --outdir outputs/exp1
```

Parameters:

* `--scenario {small,medium,custom}`: prebuilt mixes (see `benchmarks/scenarios.py`).
* `--steps`: number of simulator ticks.
* `--llm-interval`: how often LLM agents refresh intent (in ticks).
* `--mix-json`: path to a JSON that overrides the agent roster.
* `--outdir`: write Parquet/CSV and plots here.

## Notes

* This is a research simulator. **Not investment advice**.
* Microsecond-level HFT microstructure is out of scope; we target low-/mid-frequency agent studies.
* For larger runs, prefer Parquet outputs and disable plots (`--no-plots`).

License: Apache-2.0

```
```
