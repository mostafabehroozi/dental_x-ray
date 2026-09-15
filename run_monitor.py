"""Progress monitoring and failure control for every stage of a run.

The pipeline asks a model tens of questions per image, for dozens of images, for
several experiments. Two things then matter more than anything the code prints in
between:

* Nothing fails silently, and no single failure takes the sweep with it. Every
  stage (loading a dataset, running an experiment, adapting location truth,
  scoring, writing reports) wraps its items in `guard()`. A failed item prints its
  complete traceback once, is recorded in a `Ledger` with its scope, and the loop
  moves to the next item. A run that keeps failing (a dead server, a rejected key)
  stops on its own after `stop_after` consecutive failures instead of burning the
  whole dataset on the same error.
* What is printed while it runs is dense. One line per image carries the answers,
  the call and cache counts and an ETA; individual model calls are summarized, not
  listed, unless something is worth looking at (slow, truncated, empty) or CALL_LOG
  is set to "each" for debugging. Failures are the exception to density: an error,
  a traceback, and the prompt/response behind a parse failure are always printed in
  full, because a truncated failure is not evidence.

Nothing here decides anything about the science; it only decides what reaches the
console and what happens to the loop when a step raises.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

CALL_LOG = "auto"          # "auto": only calls worth looking at | "each": every call | "off": none
SLOW_CALL_SECONDS = 120.0  # a call this slow is worth a line of its own
CALL_LOG_POLICIES = ("auto", "each", "off")


# ----------------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------------
def human_time(seconds: float | None) -> str:
    """Compact duration: 8.4s, 3m12s, 1h04m."""
    if seconds is None:
        return "-"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"


def human_count(value: int | None) -> str:
    """Compact integer: 812, 12.4k, 1.2M."""
    if value is None:
        return "-"
    if abs(value) < 1000:
        return str(value)
    if abs(value) < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


def clip(text, limit: int = 160) -> str:
    """One dense line of `text`, with the number of characters left out."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else f"{flat[:limit]}...(+{len(flat) - limit} chars)"


def monitor(event: str, message: str = "", **fields) -> None:
    """Compact, consistent console event; fields that are None are left out."""
    head = f"[{event}] {message}".rstrip()
    parts = [head] + [f"{key}={value}" for key, value in fields.items() if value is not None]
    print(" | ".join(parts), flush=True)


def failure_details(prompt: str, response: str, problems=None) -> None:
    """The complete evidence behind a recoverable failure: never clipped."""
    if problems:
        print("Problems: " + "; ".join(str(p) for p in problems), flush=True)
    print("PROMPT (full):\n" + prompt, flush=True)
    print("RESPONSE (full):\n" + response, flush=True)


def error_details(exc: BaseException, *, trace: bool = False) -> str:
    """The complete error text (providers put the useful part in the message), optionally its traceback."""
    text = f"{type(exc).__name__}: {exc}"
    print("ERROR (full):\n" + text, flush=True)
    if trace:
        formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print("TRACEBACK (full):\n" + formatted.rstrip(), flush=True)
    return text


# ----------------------------------------------------------------------------
# Session tally: retries and warnings counted where they happen, printed in summaries
# ----------------------------------------------------------------------------
TALLY: dict[str, int] = {}


def tally(name: str, count: int = 1) -> None:
    TALLY[name] = TALLY.get(name, 0) + count


def tally_snapshot() -> dict[str, int]:
    return dict(TALLY)


def tally_since(snapshot: dict[str, int]) -> dict[str, int]:
    """What has been counted since `snapshot`, so one stage can report only its own events."""
    return {k: v - snapshot.get(k, 0) for k, v in TALLY.items() if v - snapshot.get(k, 0) > 0}


# ----------------------------------------------------------------------------
# Model calls: counted always, printed only when worth reading
# ----------------------------------------------------------------------------
class CallLog:
    """Counters for one model role, and the policy for printing individual calls.

    policy "auto" (the default, from CALL_LOG) prints the role's first call, so the
    console shows the model answering, then only calls worth looking at: slow,
    truncated, or empty. "each" prints every call, for debugging one image; "off"
    prints none. Either way every call is counted, and the per-image and per-stage
    summaries carry the totals.
    """

    def __init__(self, role: str, policy: str | None = None, slow_seconds: float | None = None) -> None:
        if policy is not None and policy not in CALL_LOG_POLICIES:
            raise ValueError(f"call log policy must be one of {CALL_LOG_POLICIES}, got {policy!r}")
        self.role, self._policy, self._slow = role, policy, slow_seconds
        self.requests = self.calls = self.cache_hits = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0
        self.truncated = self.empty = self.slow = 0

    @property
    def policy(self) -> str:
        return self._policy or CALL_LOG

    @property
    def slow_seconds(self) -> float:
        return self._slow if self._slow is not None else SLOW_CALL_SECONDS

    def cached(self, reply: dict) -> dict:
        """A reply served from the exact-response cache: counted, printed only under "each"."""
        self.requests += 1
        self.cache_hits += 1
        if self.policy == "each":
            monitor(f"{self.role.upper()} CACHE HIT", f"request {self.requests}",
                    in_tok=reply.get("prompt_tokens"), out_tok=reply.get("completion_tokens"),
                    finish=reply.get("finish_reason"))
        return reply

    def live(self, reply: dict) -> dict:
        """A reply from the model: counted, and printed when the policy or the reply asks for it."""
        self.requests += 1
        self.calls += 1
        self.prompt_tokens += reply.get("prompt_tokens") or 0
        self.completion_tokens += reply.get("completion_tokens") or 0
        latency = reply.get("latency_seconds") or 0.0
        self.seconds += latency
        empty = not (reply.get("text") or "").strip()
        slow = latency >= self.slow_seconds
        self.truncated += bool(reply.get("truncated"))
        self.empty += empty
        self.slow += slow
        notable = reply.get("truncated") or empty or slow
        if self.policy == "each" or (self.policy == "auto" and (notable or self.calls == 1)):
            flags = " ".join(f for f, on in (("TRUNCATED", reply.get("truncated")), ("EMPTY", empty),
                                             ("SLOW", slow)) if on)
            monitor(f"{self.role.upper()} CALL", f"call {self.calls}", took=human_time(latency),
                    in_tok=reply.get("prompt_tokens"), out_tok=reply.get("completion_tokens"),
                    finish=reply.get("finish_reason"), flags=flags or None)
        return reply

    def totals(self) -> dict:
        return {"requests": self.requests, "calls": self.calls, "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "seconds": round(self.seconds, 1), "truncated": self.truncated, "empty": self.empty,
                "slow": self.slow}

    def line(self, counts: bool = True) -> str:
        """Dense one-line state of this role. counts False leaves out the call and cache totals,
        for a caller (a Progress summary) that already prints them."""
        parts = []
        if counts:
            parts.append(f"calls={self.calls}")
            if self.cache_hits:
                parts.append(f"cache={self.cache_hits}")
        parts.append(f"tok={human_count(self.prompt_tokens)}/{human_count(self.completion_tokens)}")
        if self.calls:
            parts.append(f"avg={human_time(self.seconds / self.calls)}")
        for name, value in (("trunc", self.truncated), ("empty", self.empty), ("slow", self.slow)):
            if value:
                parts.append(f"{name}={value}")
        return " ".join(parts)


# ----------------------------------------------------------------------------
# Failure ledger and the guard that keeps a loop alive
# ----------------------------------------------------------------------------
class Ledger:
    """Every failure of one session: its scope, its complete text, and its traceback.

    A recorded failure has already been printed in full. What the ledger adds is the
    end-of-stage answer to "what did not finish, and why", and a JSON file next to
    the run so a Kaggle session that scrolled away can still be read.
    """

    def __init__(self, name: str = "run") -> None:
        self.name = name
        self.entries: list[dict] = []

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def record(self, scope: str, exc: BaseException, *, note: str | None = None) -> dict:
        """Print the complete failure (message and traceback) and keep it."""
        monitor("FAILED", scope, reason=type(exc).__name__, note=note)
        text = error_details(exc, trace=True)
        entry = {"scope": scope, "reason": type(exc).__name__, "error": text, "note": note,
                 "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                 "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.entries.append(entry)
        return entry

    def note(self, scope: str, reason: str, **fields) -> dict:
        """A problem that is not an exception (a missing input, an empty result) but still must be answered for."""
        monitor("PROBLEM", scope, reason=reason, **fields)
        entry = {"scope": scope, "reason": reason, "error": reason, "note": None, "traceback": None,
                 "fields": {k: v for k, v in fields.items() if v is not None},
                 "when": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.entries.append(entry)
        return entry

    def by_reason(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for entry in self.entries:
            grouped.setdefault(entry["reason"], []).append(entry["scope"])
        return grouped

    def save(self, path: str | Path) -> Path | None:
        """Write the failures next to the run; nothing is written when there are none."""
        if not self.entries:
            return None
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"name": self.name, "failures": self.entries}, indent=1), encoding="utf-8")
        return target

    def report(self, scopes_per_reason: int = 5, path: str | Path | None = None) -> None:
        """Dense end-of-stage answer to "what failed": one line per reason, the scopes named."""
        if not self.entries:
            monitor("HEALTH", self.name, failures=0)
            return
        monitor("HEALTH", self.name, failures=len(self.entries), reasons=len(self.by_reason()))
        for reason, scopes in sorted(self.by_reason().items(), key=lambda kv: -len(kv[1])):
            shown = ", ".join(scopes[:scopes_per_reason])
            more = f" (+{len(scopes) - scopes_per_reason} more)" if len(scopes) > scopes_per_reason else ""
            print(f"  {reason} x{len(scopes)}: {shown}{more}", flush=True)
        saved = self.save(path) if path else None
        if saved:
            print(f"  full tracebacks: {saved}", flush=True)


class Step:
    """The outcome of one guarded item: `ok` is False when it failed, `error` holds the exception."""

    __slots__ = ("scope", "ok", "error")

    def __init__(self, scope: str) -> None:
        self.scope, self.ok, self.error = scope, True, None


class guard:
    """Run one item of a loop; a failure is recorded and the loop continues.

        with mon.guard(f"{name}/{image_id}", ledger) as step:
            ...
        if not step.ok:
            continue

    Only Exception is caught, so Ctrl-C and a Kaggle session kill still stop the run
    at once. `reraise` names exception types that must propagate anyway (a corrupt
    artifact or a mixed-up output directory is a configuration error, not a bad
    image, and resuming past it would hide it).
    """

    def __init__(self, scope: str, ledger: Ledger | None = None, *, note: str | None = None,
                 reraise: tuple = ()) -> None:
        self.step = Step(scope)
        self.ledger, self.note, self.reraise = ledger, note, reraise

    def __enter__(self) -> Step:
        return self.step

    def __exit__(self, kind, exc, tb) -> bool:
        if exc is None or not isinstance(exc, Exception):
            return False
        if self.reraise and isinstance(exc, self.reraise):
            return False
        self.step.ok, self.step.error = False, exc
        if self.ledger is not None:
            self.ledger.record(self.step.scope, exc, note=self.note)
        else:
            monitor("FAILED", self.step.scope, reason=type(exc).__name__, note=self.note)
            error_details(exc, trace=True)
        return True


# ----------------------------------------------------------------------------
# Progress over a dataset: one dense line per item, then one dense summary
# ----------------------------------------------------------------------------
class Progress:
    """One dense line per finished item, with running totals and an ETA.

    `item()` is the line for a finished item: its own detail string (what the model
    answered) plus whatever counters the caller passes, which are also summed for
    `done()`. `skip()` counts an item resumed from disk without printing, and
    `failure()` counts a failed one (it has already printed its traceback) and keeps
    the consecutive-failure count that lets a caller stop a hopeless run.
    """

    def __init__(self, total: int, label: str = "", unit: str = "item", every: int = 1) -> None:
        self.total, self.label, self.unit, self.every = total, label, unit, max(1, every)
        self.started = time.perf_counter()
        self.index = self.finished = self.skipped = self.failed = 0
        self.consecutive_failures = 0
        self.counters: dict[str, int] = {}
        self.tally_at_start = tally_snapshot()
        self.last_item_at = self.started

    def _advance(self) -> float:
        self.index += 1
        now = time.perf_counter()
        elapsed, self.last_item_at = now - self.last_item_at, now
        return elapsed

    def _eta(self) -> str | None:
        done = self.finished + self.failed
        if not done or self.index >= self.total:
            return None
        rate = (time.perf_counter() - self.started) / done
        return human_time(rate * (self.total - self.index))

    def add(self, **counters) -> None:
        for key, value in counters.items():
            if value:
                self.counters[key] = self.counters.get(key, 0) + value

    def skip(self, name: str = "") -> None:
        self.index += 1
        self.skipped += 1
        self.last_item_at = time.perf_counter()

    def item(self, name: str, detail: str = "", **counters) -> None:
        elapsed = self._advance()
        self.finished += 1
        self.consecutive_failures = 0
        self.add(**counters)
        fields = " | ".join(str(p) for p in (detail, *(f"{k}={v}" for k, v in counters.items() if v is not None)) if p)
        eta = self._eta()
        print(f"[{self.index:>{len(str(self.total))}}/{self.total}] {name} | {human_time(elapsed)}"
              + (f" | {fields}" if fields else "") + (f" | left={eta}" if eta else ""), flush=True)

    def failure(self, name: str = "") -> int:
        self._advance()
        self.failed += 1
        self.consecutive_failures += 1
        return self.consecutive_failures

    def stop(self, reason: str) -> None:
        monitor("STOPPED", self.label, reason=reason, done=self.index, of=self.total)

    def summary(self) -> dict:
        return {"label": self.label, "total": self.total, "done": self.finished, "resumed": self.skipped,
                "failed": self.failed, "seconds": round(time.perf_counter() - self.started, 1),
                **self.counters, **tally_since(self.tally_at_start)}

    def done(self, detail: str = "", **extra) -> dict:
        """One dense closing line: how long, how many, and every counter that is not zero."""
        summary = self.summary()
        summary.update({k: v for k, v in extra.items() if v is not None})
        seconds = summary["seconds"]
        per = seconds / summary["done"] if summary["done"] else None
        parts = [f"{self.total} {self.unit}{'s' if self.total != 1 else ''} in {human_time(seconds)}"
                 + (f" ({human_time(per)}/{self.unit})" if per else ""),
                 f"done={summary['done']} resumed={summary['resumed']} failed={summary['failed']}"]
        skip = {"label", "total", "done", "resumed", "failed", "seconds"}
        rest = " ".join(f"{k}={human_count(v) if isinstance(v, int) else v}"
                        for k, v in summary.items() if k not in skip and v)
        if rest:
            parts.append(rest)
        if detail:
            parts.append(detail)
        print(f"[DONE] {self.label or self.unit} | " + " | ".join(parts), flush=True)
        return summary
