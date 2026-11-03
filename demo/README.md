# Demo: single agent vs. a manager-led team, at a matched budget

New to the demo? Start with `docs/presenter-demo-background-knowledge.md` (HotpotQA, question types, the search tool, what k means, how the team's tools work).

The suggested seminar demo: compare **one general agent** with a **manager + researcher + analyst + critic** team on the same questions, **controlling for model calls and total tokens**.

| Condition | What runs | Budget |
|---|---|---|
| **A. Single** | One agent with a `search` tool answers once | ~1 try |
| **B. Vote@k** | The same agent answers k times; majority vote (self-consistency) | k chosen so tokens ≈ team |
| **C. Team** | Manager calls `research`, `analyze`, `critique`, which are three specialist agents exposed **as tools** (OpenAI Agents SDK `agent.as_tool`). The manager writes the final answer | whatever it uses |

Everything else is identical: the model (Qwen2.5-7B-Instruct on one A10 via vLLM), temperature 0.7, the search tool, the questions, and the per-call output cap.

## Files

| File | Purpose |
|---|---|
| `select_questions.py` | Picks 5 experiment questions (3 bridge, 2 comparison) + 15 extra UI questions from the HotpotQA dev set by a seeded rule, plus 3 hand-picked live-demo questions listed first (`DEMO_PICKS`, see `docs/demo-script.md`) → `data/questions.json`; also writes `data/corpus.json` (all 66,581 dev-set paragraphs, the search pool for custom questions; not in git) |
| `mas_demo.py` | The agents, the BM25 `search` tool, call/token counting (`CountingModel`), scoring, voting |
| `make_sample_articles.py` | Copies five linked articles from `data/corpus.json` into `data/sample_articles.json`, plus three example custom questions with gold answers |
| `run_experiment.py` | Runs A/B/C over the 5 experiment questions × repeats → `results/` |
| `analyze.py` | Table, plot and example trace of the pre-computed experiment (backup material for Q&A) |
| `app.py` | Side-by-side web UI (Streamlit): pick one of 23 questions or type your own, pick two setups, press Run |
| `run_demo.pbs` | PBS job: start vLLM on an A10, then run the experiment (default) or serve the UI (`MODE=ui`) |

## How to run

The environment is a uv venv (vllm 0.32, openai-agents 0.24, streamlit 1.66, Python 3.12). Set its path (`VENV`) and the model weights path (`MODEL`) at the top of `run_demo.pbs`, and the `group_list` (X) and email (Y) in its `#PBS` header.

```bash
cd demo
python select_questions.py        # once (login node is fine): data/questions.json + data/corpus.json
python make_sample_articles.py    # data/sample_articles.json (five articles for custom questions)
qsub -v MODE=smoke run_demo.pbs   # ~5 min: 2 questions × 1 repeat, k=3 → results/smoke/
qsub run_demo.pbs                 # full run (5 questions × 5 repeats) + analyze.py
```

Side-by-side web UI (recommended for the talk):

```bash
qsub -v MODE=ui run_demo.pbs      # vLLM + Streamlit on one A10; wait ~3-5 min
cat logs/ui_tunnel.txt            # prints the exact ssh command for your node
```

On your **local** machine, run the printed command, e.g.
`ssh -N -L 8501:gpu0025.prime.ccast:8501 <username>@prime.ccast.ndsu.edu`,
then open <http://localhost:8501>. Pick a question (1-3: the hand-picked questions used in the live-demo script; 4-8: the pre-computed experiment set; 9-23: extra HotpotQA questions; last entry: type your own question and gold answer, searched over the 66,581-paragraph collection; `data/sample_articles.json` holds five linked articles and three example questions to type in) and a setup for each side (default: Single agent vs. Team; switch one side to "× k tries, majority vote" for the fair comparison; the k box only applies to that setup) and press **Run**. The left setup runs first and appears all at once, then the right one. Each panel shows the answer (✅/❌ against the gold answer), calls, tokens, seconds, a table of every model call, and each call's full input messages and output (hide or show them under "Display options"). Stop the job with `qdel <jobid>` when done. A run takes about 4 s (single), 12 s (team) or 15–30 s (vote of 4).

## How the counting works

Every agent's model is a `CountingModel`, a subclass of the SDK's `OpenAIChatCompletionsModel` that appends one record per model call (agent, input tokens, output tokens, tool calls, text) to a per-run log. Because the specialists' nested runs use the same model class, their calls are included in the team's totals. Token numbers come from vLLM's `usage` field.

The k samples for condition B include condition A's single try, so A costs nothing extra. k = round(mean team tokens ÷ mean single-try tokens), clamped to [3, 15], unless `--k` is given.

## Caveats

- 5 questions × 5 repeats is a **demonstration, not a benchmark**
- HotpotQA may be partly memorized by the model. The prompts tell every condition to answer from the documents
- Tool-call formatting errors or `max_turns` overruns count as wrong answers (column `errors` in `results.csv`)
