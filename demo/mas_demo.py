"""Single agent vs. manager-led team, on HotpotQA, at a matched budget.

Three conditions, all on the SAME local model, tools, temperature and questions:
  A  single      one agent with a `search` tool answers once
  B  vote@k      the same single agent answers k times; majority vote
                 (self-consistency, the budget-matched baseline; k is chosen
                 so B spends about as many tokens as C)
  C  team        Manager + Researcher + Analyst + Critic, with the three
                 specialists exposed to the manager as tools
                 (OpenAI Agents SDK "agents as tools"; the manager keeps
                 responsibility for the final answer)

Every model call is counted at the model layer (`CountingModel`), so the
specialists' nested runs are included in the team's call and token totals.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
import os
import re
import string
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from agents import (
    Agent,
    ModelSettings,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.models.chatcmpl_converter import Converter

HERE = Path(__file__).resolve().parent
set_tracing_disabled(True)  # no OpenAI key / no trace upload: everything stays local

# --------------------------------------------------------------------------------------
# Configuration (overridable through environment variables)
# --------------------------------------------------------------------------------------
BASE_URL = os.environ.get("MAS_BASE_URL")  # e.g. http://gpu0025:8000/v1
MODEL_NAME = os.environ.get("MAS_MODEL", "qwen2.5-7b-instruct")
TEMPERATURE = float(os.environ.get("MAS_TEMPERATURE", "0.7"))  # same for every condition
MAX_TOKENS_PER_CALL = int(os.environ.get("MAS_MAX_TOKENS", "768"))
SEARCH_TOP_K = 3


def resolve_base_url() -> str:
    """Use $MAS_BASE_URL, else the address file written by the vLLM job."""
    if BASE_URL:
        return BASE_URL
    addr_file = HERE / "logs" / "server_address"
    if addr_file.exists():
        return addr_file.read_text().strip()
    return "http://localhost:8000/v1"


# --------------------------------------------------------------------------------------
# Data and the per-question document store (the 10 HotpotQA paragraphs)
# --------------------------------------------------------------------------------------
def load_questions(path: Path | None = None, experiment_only: bool = False) -> list[dict]:
    """All demo questions; experiment_only=True gives the 5 behind the pre-computed results."""
    qs = json.loads((path or HERE / "data" / "questions.json").read_text())
    return [q for q in qs if q.get("experiment", True)] if experiment_only else qs


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of in on at to for by with and or is was were be who what which when where how did does do".split())


def _tok(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOP]


class DocStore:
    """Tiny BM25 index over one question's paragraphs. No external services."""

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.toks = [_tok(d["title"] + " " + d["title"] + " " + d["text"]) for d in docs]
        self.avgdl = sum(map(len, self.toks)) / max(1, len(self.toks))
        df = Counter(t for ts in self.toks for t in set(ts))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = SEARCH_TOP_K) -> list[dict]:
        q = _tok(query)
        scores = []
        for i, ts in enumerate(self.toks):
            tf = Counter(ts)
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(ts) / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] for s, i in scores[:k] if s > 0]


class CorpusStore:
    """BM25 over every distinct paragraph in the HotpotQA dev set (~66,600 Wikipedia intros).

    Used for custom questions typed into the UI. Same scoring as DocStore, but with an
    inverted index so a query over 66k paragraphs takes milliseconds.
    """

    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lens = []
        for i, d in enumerate(docs):
            ts = _tok(d["title"] + " " + d["title"] + " " + d["text"])
            self.lens.append(len(ts))
            for t, f in Counter(ts).items():
                self.postings.setdefault(t, []).append((i, f))
        n = len(docs)
        self.avgdl = sum(self.lens) / max(1, n)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, k: int = SEARCH_TOP_K) -> list[dict]:
        scores: Counter = Counter()
        for t in set(_tok(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                scores[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl))
        return [self.docs[i] for i, _ in scores.most_common(k)]


_CORPUS: CorpusStore | None = None


def get_corpus() -> CorpusStore:
    """Load data/corpus.json (written by select_questions.py) once per process."""
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = CorpusStore(json.loads((HERE / "data" / "corpus.json").read_text()))
    return _CORPUS


def store_for(q: dict) -> DocStore | CorpusStore:
    """A HotpotQA question searches its own 10 paragraphs; a custom question searches the corpus."""
    return DocStore(q["docs"]) if q.get("docs") else get_corpus()


@dataclass
class QCtx:
    """Run context shared by every agent (and nested agent-tool run) for one question."""

    store: DocStore | CorpusStore
    searches: list[str] = field(default_factory=list)


@function_tool
def search(ctx: RunContextWrapper[QCtx], query: str) -> str:
    """Search the document collection. Returns the 3 most relevant paragraphs, each with its title.

    Args:
        query: keywords or a short question, e.g. "Manning Field" or "Robert Jordan born".
    """
    ctx.context.searches.append(query)
    hits = ctx.context.store.search(query)
    if not hits:
        return "No matching documents."
    return "\n\n".join(f"[{d['title']}] {d['text']}" for d in hits)


# --------------------------------------------------------------------------------------
# Call/token accounting at the model layer
# --------------------------------------------------------------------------------------
_CALL_LOG: contextvars.ContextVar[list | None] = contextvars.ContextVar("call_log", default=None)


def _summarize_output(output: list[Any]) -> tuple[list[dict], str]:
    tool_calls, texts = [], []
    for item in output:
        name = getattr(item, "name", None)
        if name is not None and hasattr(item, "arguments"):
            tool_calls.append({"tool": name, "arguments": item.arguments})
        for part in getattr(item, "content", None) or []:
            if getattr(part, "text", None):
                texts.append(part.text)
    return tool_calls, "\n".join(texts)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # list of content parts
        return "\n".join(str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _readable_messages(system: str | None, items: Any) -> list[dict]:
    """The chat messages this call sends to the model, as [{role, text}] for display.

    Built with the SDK's own converter, so it matches what goes over the wire (the tool
    definitions are sent separately; vLLM adds them to the prompt, and they count as input tokens).
    """
    msgs = Converter.items_to_messages(items)
    if system:
        msgs.insert(0, {"role": "system", "content": system})
    names: dict[str, str] = {}
    out = []
    for m in msgs:
        role = m.get("role")
        if role == "system":
            out.append({"role": "system", "text": _content_text(m.get("content"))})
        elif role == "user":
            out.append({"role": "user", "text": _content_text(m.get("content"))})
        elif role == "assistant":
            parts = [_content_text(m.get("content"))] if m.get("content") else []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                names[tc.get("id", "")] = fn.get("name", "?")
                parts.append(f"→ calls {fn.get('name')}({fn.get('arguments')})")
            out.append({"role": "assistant", "text": "\n".join(p for p in parts if p)})
        elif role == "tool":
            out.append({"role": f"tool result: {names.get(m.get('tool_call_id', ''), 'tool')}",
                        "text": _content_text(m.get("content"))})
        else:
            out.append({"role": str(role), "text": _content_text(m.get("content"))})
    return out


class CountingModel(OpenAIChatCompletionsModel):
    """Chat Completions model that logs every call (who, input messages, tokens, tool calls, text)."""

    def __init__(self, label: str, **kw):
        super().__init__(**kw)
        self.label = label

    async def get_response(self, system_instructions, input, model_settings, tools, *args, **kwargs):
        t0 = time.perf_counter()
        resp = await super().get_response(system_instructions, input, model_settings, tools, *args, **kwargs)
        tool_calls, text = _summarize_output(resp.output)
        try:
            messages = _readable_messages(system_instructions, input)
        except Exception as e:  # display only; never break a run over it
            messages = [{"role": "note", "text": f"(could not render the input: {e})"}]
        rec = {
            "agent": self.label,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "seconds": round(time.perf_counter() - t0, 2),
            "tool_calls": tool_calls,
            "text": text,
            "tools_offered": [{"name": t.name, "description": getattr(t, "description", "")} for t in tools or []],
            "messages": messages,
        }
        log = _CALL_LOG.get()
        if log is not None:
            log.append(rec)
        return resp


def _short(s: str, n: int = 80) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------------------
# Agents
# --------------------------------------------------------------------------------------
FINAL_RULE = (
    "End your reply with one line of the form 'FINAL ANSWER: <answer>' where <answer> is as short "
    "as possible (a name, a year, a few words), with no explanation."
)

SINGLE_INSTRUCTIONS = f"""You answer multi-hop questions using a document collection.
Use the `search` tool to find evidence; most questions need two searches (find the first entity,
then search for the fact about it). Do not rely on memory: base the answer on the documents.
Think briefly, then answer. {FINAL_RULE}"""

RESEARCHER_INSTRUCTIONS = """You are the RESEARCHER on a team. You receive a research request.
Use the `search` tool (as many times as needed, at most 4) to find the passages that answer it.
Most requests need two searches (find the first entity, then search for the fact about it).
Reply with the relevant facts only, each followed by its document title in brackets, quoting exact
names, dates and numbers. Do not answer the overall question; just report evidence."""

ANALYST_INSTRUCTIONS = """You are the ANALYST on a team. You receive a question and evidence.
Reason step by step using ONLY the evidence: identify the bridge entity, compare dates or numbers,
and work out what exactly is being asked. Keep it to at most 6 short lines and end with
'PROPOSED ANSWER: <short answer>'. If the evidence is insufficient, say exactly what is missing."""

CRITIC_INSTRUCTIONS = """You are the CRITIC on a team. You receive a question, evidence and a
proposed answer. Check: (1) every step is supported by the evidence; (2) the answer is the type of
thing asked for (e.g., a year, a person, a former name) and answers exactly the question;
(3) it is short. Reply 'APPROVE' if all checks pass, otherwise 'REVISE: <problem> -> <corrected answer>'.
Do not invent facts that are not in the evidence."""

MANAGER_INSTRUCTIONS = f"""You are the MANAGER of a research team and you are responsible for the
final answer. You cannot search yourself; use your tools:
  - research: ask the researcher to find evidence (call it again for a second hop if needed)
  - analyze: give the analyst the question and ALL the evidence; get reasoning + proposed answer
  - critique: give the critic the question, the evidence and the proposed answer
Workflow: research -> analyze -> critique. If the critic says REVISE, fix the answer (research
again if evidence is missing). Pass full text to your tools; they cannot see this conversation.
{FINAL_RULE}"""


def _settings() -> ModelSettings:
    return ModelSettings(temperature=TEMPERATURE, max_tokens=MAX_TOKENS_PER_CALL)


def _model(label: str, client: AsyncOpenAI) -> CountingModel:
    return CountingModel(label=label, model=MODEL_NAME, openai_client=client)


def build_single(client: AsyncOpenAI) -> Agent[QCtx]:
    return Agent[QCtx](
        name="Single",
        instructions=SINGLE_INSTRUCTIONS,
        tools=[search],
        model=_model("Single", client),
        model_settings=_settings(),
    )


def build_team(client: AsyncOpenAI) -> Agent[QCtx]:
    def specialist(name: str, instructions: str, tools: list) -> Agent[QCtx]:
        return Agent[QCtx](
            name=name, instructions=instructions, tools=tools,
            model=_model(name, client), model_settings=_settings(),
        )

    researcher = specialist("Researcher", RESEARCHER_INSTRUCTIONS, [search])
    analyst = specialist("Analyst", ANALYST_INSTRUCTIONS, [])
    critic = specialist("Critic", CRITIC_INSTRUCTIONS, [])
    return Agent[QCtx](
        name="Manager",
        instructions=MANAGER_INSTRUCTIONS,
        tools=[
            researcher.as_tool("research", "Find evidence in the documents. Input: what to look up.", max_turns=6),
            analyst.as_tool("analyze", "Reason over evidence. Input: the question plus all evidence text.", max_turns=2),
            critic.as_tool("critique", "Check a proposed answer. Input: question, evidence, proposed answer.", max_turns=2),
        ],
        model=_model("Manager", client),
        model_settings=_settings(),
    )


def make_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=resolve_base_url(), api_key="EMPTY", timeout=300, max_retries=2)


# --------------------------------------------------------------------------------------
# Running one agent on one question
# --------------------------------------------------------------------------------------
_FINAL_RE = re.compile(r"FINAL ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)


def extract_answer(text: str) -> str:
    m = _FINAL_RE.findall(text or "")
    ans = m[-1] if m else next((ln for ln in reversed((text or "").splitlines()) if ln.strip()), "")
    return ans.strip().strip("*").strip().strip('"').strip("'").rstrip(".").strip()


async def run_agent(agent: Agent[QCtx], q: dict, max_turns: int) -> dict:
    """Run `agent` on question `q`; return answer + all model calls made (incl. nested)."""
    log: list[dict] = []
    _CALL_LOG.set(log)
    ctx = QCtx(store=store_for(q))
    t0 = time.perf_counter()
    error = None
    try:
        result = await Runner.run(agent, q["question"], context=ctx, max_turns=max_turns)
        final_text = str(result.final_output)
    except Exception as e:  # max-turns exceeded, malformed tool call, server error ...
        final_text, error = "", f"{type(e).__name__}: {e}"[:500]
    pred = extract_answer(final_text)
    return {
        "prediction": pred,
        "final_text": final_text[:3000],
        "error": error,
        "calls": len(log),
        "input_tokens": sum(c["input_tokens"] for c in log),
        "output_tokens": sum(c["output_tokens"] for c in log),
        "total_tokens": sum(c["input_tokens"] + c["output_tokens"] for c in log),
        "seconds": round(time.perf_counter() - t0, 2),
        "searches": list(ctx.searches),
        "trace": log,
    }


# --------------------------------------------------------------------------------------
# Scoring (HotpotQA-style EM/F1, plus Unicode folding so "ibnʿArabī" == "Ibn Arabi")
# --------------------------------------------------------------------------------------
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    # keep letters/digits; modifier letters such as "ʿ" (category Lm) become spaces
    s = "".join(ch if ch.isalnum() and unicodedata.category(ch) != "Lm" else " " for ch in s)
    return " ".join(w for w in s.split() if w not in {"a", "an", "the"})


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold) and normalize(gold) != "")


def f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return round(2 * prec * rec / (prec + rec), 3)


def majority_vote(preds: list[str]) -> str:
    """Most common normalized answer (ignoring empty ones); ties go to the earliest."""
    norm = [normalize(p) for p in preds]
    counts = Counter(n for n in norm if n)
    if not counts:
        return ""
    best = max(counts.values())
    for p, n in zip(preds, norm):
        if n and counts[n] == best:
            return p
    return ""


async def gather_limited(coros, limit: int):
    sem = asyncio.Semaphore(limit)

    async def one(c):
        async with sem:
            return await c

    return await asyncio.gather(*(one(c) for c in coros))


# --------------------------------------------------------------------------------------
# One setup on one question (used by the UI)
# --------------------------------------------------------------------------------------
SETUPS = {
    "single": "Single agent (1 try)",
    "vote": "Single agent × k tries, majority vote",
    "team": "Team: manager + researcher, analyst, critic (as tools)",
}


async def run_setup(setup: str, q: dict, k: int = 4) -> dict:
    """Run one of SETUPS on question q. Returns answer, totals and every model call."""
    client = make_client()  # a fresh client per event loop (the UI calls asyncio.run per click)
    t0 = time.perf_counter()
    if setup == "team":
        runs = [await run_agent(build_team(client), q, max_turns=12)]
    elif setup == "single":
        runs = [await run_agent(build_single(client), q, max_turns=10)]
    elif setup == "vote":
        agent = build_single(client)
        runs = list(await asyncio.gather(*(run_agent(agent, q, max_turns=10) for _ in range(k))))
    else:
        raise ValueError(setup)
    answer = majority_vote([r["prediction"] for r in runs]) if setup == "vote" else runs[0]["prediction"]
    calls = [dict(c, attempt=i + 1) for i, r in enumerate(runs) for c in r["trace"]]
    return {
        "setup": setup,
        "answer": answer,
        "correct": bool(exact_match(answer, q["answer"])),
        "attempt_answers": [r["prediction"] for r in runs],
        "errors": [r["error"] for r in runs if r["error"]],
        "calls": calls,
        "n_calls": len(calls),
        "tokens": sum(c["input_tokens"] + c["output_tokens"] for c in calls),
        "seconds": round(time.perf_counter() - t0, 1),
    }
