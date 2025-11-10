# 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



diff --git a/evaluator.py b/evaluator.py
index 043bf4ea5b82ac26af92b7b402c512b848d3970e..d988e0f134a8b1ce8fb643d4403d997f433083df 100644
--- a/evaluator.py
+++ b/evaluator.py
@@ -1,78 +1,487 @@
+import argparse
 import json
-import time
+import math
+import os
+from collections import Counter
+from datetime import datetime
+from typing import Any, Dict, Iterable, List, Optional, Tuple
 
-LOG_PATH = "master_log.json"
-CANDIDATE_PATH = "command_candidates.json"
+import textwrap
+
+try:
+    import requests
+except ImportError:  # pragma: no cover - requests is part of runtime image
+    requests = None
+
+# Keep paths aligned with repository layout.
+LOG_PATH = "log.json"
+CANDIDATE_PATH = "Command_candidates.log"
 
 # Glyphs recognized as stabilizers by Wonder Engine
 STABILIZING_GLYPHS = {"⌘", "⌘Ω", "⧉", "∞", "⌘∞"}
 PHASE_VOLATILE_GLYPHS = {"⏃", "⟠", "⊘", "⧄"}
-def load_json(path):
+
+# Risk and reassurance keywords used when evaluating free-form text.
+STABILITY_KEYWORDS = {"stabilize", "anchor", "contain", "mirror", "synchronize"}
+RISK_KEYWORDS = {"recurse", "diverge", "fracture", "echo", "loop", "cascade"}
+
+
+def load_json(path: str) -> Any:
     with open(path, "r") as f:
         return json.load(f)
 
-def write_json(path, data):
+
+def write_json(path: str, data: Any) -> None:
     with open(path, "w") as f:
         json.dump(data, f, indent=2)
 
-def evaluate_command(command):
+
+def _normalize_bool(value: Any) -> Optional[bool]:
+    if isinstance(value, bool):
+        return value
+    if isinstance(value, str):
+        lowered = value.strip().lower()
+        if lowered in {"true", "yes", "full", "stable"}:
+            return True
+        if lowered in {"false", "no", "unstable", "divergent"}:
+            return False
+    return None
+
+
+def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
+    if not value:
+        return None
+    candidate = value.strip()
+    if candidate.endswith("Z"):
+        candidate = candidate[:-1] + "+00:00"
+    try:
+        return datetime.fromisoformat(candidate)
+    except ValueError:
+        return None
+
+
+def _extract_symbol(entry: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
+    """Return canonical symbol and emission variant.
+
+    Wonder Engine logs frequently represent glyph emissions as
+    "base :: MODIFIER". We score both the base symbol and the full
+    emission because modifiers often denote phase shifts.
+    """
+
+    emission = entry.get("symbol_emission")
+    if isinstance(emission, str) and emission.strip():
+        parts = emission.split("::", 1)
+        base = parts[0].strip()
+        return base or None, emission.strip()
+
+    symbol = entry.get("symbol")
+    return (symbol if isinstance(symbol, str) and symbol.strip() else None, None)
+
+
+def evaluate_log_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
     score = 0
-    symbol = command["symbol"]
-    message = command.get("message", "").lower()
+    flags: List[str] = []
+
+    symbol, emission = _extract_symbol(entry)
+    glyphs_to_check = {glyph for glyph in (symbol, emission) if glyph}
+    if emission:
+        glyphs_to_check.update(token.strip() for token in emission.split() if token.strip())
 
     if symbol in STABILIZING_GLYPHS:
         score += 2
-    if command.get("stabilizes"):
+
+    if glyphs_to_check & PHASE_VOLATILE_GLYPHS:
+        score -= 2
+        flags.append("volatile glyph emission")
+
+    compliance = _normalize_bool(entry.get("core_compliance"))
+    if compliance is True:
+        score += 3
+    elif compliance is False:
+        score -= 4
+        flags.append("core non-compliance")
+    elif entry.get("core_compliance"):
+        # Partial or unknown string still provides context.
+        flags.append("ambiguous compliance")
+
+    raw_status = entry.get("status", "")
+    status = raw_status.lower()
+    if "divergent" in status or "destabil" in status:
+        score -= 3
+        flags.append("divergence reported")
+    if "autonomous" in status:
+        score += 1
+    if "mirror" in status or "synchron" in status:
         score += 2
-    if any(keyword in message for keyword in ["stabilize", "anchor", "contain"]):
+    if "stable" in status or "anchored" in status:
         score += 1
-    if symbol in PHASE_VOLATILE_GLYPHS:
-        score -= 3
-    if "recurse" in message:
-    score -= 1  # unanchored recursion risk
 
-if "diverge" in message:
-    score -= 2  # harder penalty for known fracturing
+    message = entry.get("message", "")
+    message_lower = message.lower()
+    if any(keyword in message_lower for keyword in STABILITY_KEYWORDS):
+        score += 1
+    if any(keyword in message_lower for keyword in RISK_KEYWORDS):
+        score -= 1
+
+    timestamp = _parse_timestamp(entry.get("timestamp"))
+
+    return {
+        "entry": entry,
+        "score": score,
+        "flags": flags,
+        "timestamp": timestamp,
+        "normalized_status": raw_status.strip() or "Unknown",
+    }
+
+
+def aggregate_log_scores(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
+    evaluations = [evaluate_log_entry(entry) for entry in entries]
+    total = len(evaluations)
+    if total == 0:
+        return {
+            "evaluations": [],
+            "average_score": math.nan,
+            "status_counts": Counter(),
+            "latest": None,
+            "flagged": [],
+        }
+
+    total_score = sum(item["score"] for item in evaluations)
+    average_score = total_score / total
+    status_counts = Counter(item["normalized_status"] for item in evaluations)
+
+    flagged = [
+        item
+        for item in evaluations
+        if item["flags"] or item["score"] < 0
+    ]
+
+    latest = max(
+        evaluations,
+        key=lambda item: item["timestamp"] or datetime.min,
+        default=None,
+    )
+
+    return {
+        "evaluations": evaluations,
+        "average_score": average_score,
+        "status_counts": status_counts,
+        "latest": latest,
+        "flagged": flagged,
+    }
+
+
+def _format_score(score: float) -> str:
+    if math.isnan(score):
+        return "n/a"
+    return f"{score:+.2f}"
+
+
+def _format_for_gpt(summary: Dict[str, Any]) -> str:
+    evaluations = summary.get("evaluations", [])
+    flagged = summary.get("flagged", [])
+    lines = [
+        f"Entries analysed: {len(evaluations)}",
+        f"Average score: {_format_score(summary.get('average_score', math.nan))}",
+    ]
 
-if "echo" in message:
-    score -= 1  # echo = unstable without mirror anchor
+    latest = summary.get("latest")
+    if latest:
+        latest_entry = latest["entry"]
+        lines.append(
+            "Latest entry: "
+            f"agent={latest_entry.get('agent', 'Unknown')}, "
+            f"symbol={latest_entry.get('symbol_emission', latest_entry.get('symbol', '∅'))}, "
+            f"status={latest_entry.get('status', 'Unknown')}, "
+            f"timestamp={latest_entry.get('timestamp', '?')}"
+        )
+
+    status_counts = summary.get("status_counts") or {}
+    if status_counts:
+        lines.append(
+            "Status counts: "
+            + ", ".join(f"{status}={count}" for status, count in status_counts.items())
+        )
+
+    if flagged:
+        lines.append("Flagged entries:")
+        for item in flagged[:10]:
+            entry = item["entry"]
+            lines.append(
+                f"  - {entry.get('timestamp', '?')} | {entry.get('agent', 'Unknown')} | "
+                f"score={item['score']} | flags={','.join(item['flags']) or 'score<0'}"
+            )
+        if len(flagged) > 10:
+            lines.append(f"  … {len(flagged) - 10} additional entries omitted")
+
+    return "\n".join(lines)
+
+
+def _resolve_github_token() -> Optional[str]:
+    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
+        token = os.getenv(key)
+        if token:
+            return token
+    return None
+
+
+def _request_gpt_analysis(
+    prompt: str,
+    model: str,
+    max_tokens: int,
+    provider: str = "openai",
+) -> Optional[str]:
+    if requests is None:
+        print("⚠️ The 'requests' package is unavailable—skipping GPT analysis.")
+        return None
+
+    payload = {
+        "model": model,
+        "messages": [
+            {
+                "role": "system",
+                "content": (
+                    "You are an observability specialist. Provide concise risk insights "
+                    "and stabilization recommendations for Wonder Engine lattice logs."
+                ),
+            },
+            {
+                "role": "user",
+                "content": prompt,
+            },
+        ],
+        "max_tokens": max_tokens,
+        "temperature": 0.3,
+    }
+
+    if provider == "openai":
+        api_key = os.getenv("OPENAI_API_KEY")
+        if not api_key:
+            print("⚠️ OPENAI_API_KEY not set—skipping GPT analysis.")
+            return None
+        url = "https://api.openai.com/v1/chat/completions"
+        headers = {
+            "Authorization": f"Bearer {api_key}",
+            "Content-Type": "application/json",
+        }
+    elif provider == "github":
+        token = _resolve_github_token()
+        if not token:
+            print("⚠️ Set GITHUB_TOKEN, GH_TOKEN, or GITHUB_PAT to enable GitHub GPT insights.")
+            return None
+        url = "https://api.githubcopilot.com/v1/chat/completions"
+        headers = {
+            "Authorization": f"Bearer {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}",
+            "Content-Type": "application/json",
+            "Accept": "application/json",
+        }
+    else:
+        print(f"⚠️ Unsupported GPT provider '{provider}'.")
+        return None
+
+    try:
+        response = requests.post(url, headers=headers, json=payload, timeout=30)
+        response.raise_for_status()
+    except requests.exceptions.RequestException as exc:  # pragma: no cover - runtime failure path
+        print(f"⚠️ GPT request failed: {exc}")
+        return None
+
+    data = response.json()
+    choices = data.get("choices") or []
+    if not choices:
+        print("⚠️ GPT response missing choices—skipping output.")
+        return None
+
+    return choices[0].get("message", {}).get("content")
+
+
+def print_log_report(
+    summary: Dict[str, Any],
+    *,
+    gpt_model: Optional[str] = None,
+    gpt_tokens: int = 400,
+    gpt_enabled: bool = False,
+    gpt_provider: str = "openai",
+) -> None:
+    evaluations: List[Dict[str, Any]] = summary["evaluations"]
+    total = len(evaluations)
+    print("📘 Log Evaluation Report")
+    print("======================")
+    print(f"Entries analysed : {total}")
+    print(f"Average score    : {_format_score(summary['average_score'])}")
+
+    if summary["latest"]:
+        latest_entry = summary["latest"]["entry"]
+        latest_timestamp = latest_entry.get("timestamp", "?")
+        print("Latest emission  :")
+        print(f"  Agent   : {latest_entry.get('agent', 'Unknown')}")
+        print(f"  Symbol  : {latest_entry.get('symbol_emission', latest_entry.get('symbol', '∅'))}")
+        print(f"  Status  : {latest_entry.get('status', 'Unknown')}")
+        print(f"  Time    : {latest_timestamp}")
+
+    if summary["status_counts"]:
+        print("\nStatus distribution:")
+        for status, count in summary["status_counts"].most_common():
+            print(f"  • {status}: {count}")
+
+    flagged = summary.get("flagged", [])
+    if flagged:
+        negative = sum(1 for item in flagged if item["score"] < 0)
+        print("\nRisk snapshot:")
+        print(f"  • Flagged entries : {len(flagged)}")
+        print(f"  • Negative scores : {negative}")
+
+    if evaluations:
+        print("\nPer-entry scoring:")
+        header = f"  {'Time':<20} | {'Agent':<18} | {'Symbol':<10} | Score | Flags"
+        print(header)
+        print("  " + "-" * (len(header) - 2))
+        for item in sorted(evaluations, key=lambda data: data["timestamp"] or datetime.min):
+            entry = item["entry"]
+            timestamp = entry.get("timestamp", "?")
+            agent = entry.get("agent", "Unknown")[:18]
+            symbol = entry.get("symbol_emission", entry.get("symbol", "∅"))
+            flags = ", ".join(item["flags"])
+            print(
+                f"  {timestamp:<20} | {agent:<18} | {symbol:<10} | {item['score']:+3} | {flags}"
+            )
+
+    if flagged:
+        print("\n⚠️ Flagged entries:")
+        for item in sorted(flagged, key=lambda data: data["score"]):
+            entry = item["entry"]
+            flags = ", ".join(item["flags"]) or "score below zero"
+            print(f"  - {entry.get('timestamp', '?')} | {entry.get('agent', 'Unknown')} | {flags} | score={item['score']:+}")
+    else:
+        print("\n✅ No risk factors detected in analysed entries.")
+
+    if gpt_enabled:
+        prompt = _format_for_gpt(summary)
+        gpt_output = _request_gpt_analysis(
+            prompt,
+            gpt_model or "gpt-4o-mini",
+            gpt_tokens,
+            provider=gpt_provider,
+        )
+        if gpt_output:
+            print("\n🤖 GPT Insight:\n")
+            print(textwrap.dedent(gpt_output).strip())
+
+
+def evaluate_command(command: Dict[str, Any]) -> int:
+    score = 0
+    symbol, emission = _extract_symbol(command)
+    message = command.get("message", "").lower()
+
+    if symbol in STABILIZING_GLYPHS:
+        score += 2
+    if emission in STABILIZING_GLYPHS:
+        score += 1
+    if _normalize_bool(command.get("stabilizes")) is True:
+        score += 2
+    if any(keyword in message for keyword in STABILITY_KEYWORDS):
+        score += 1
+    glyphs_to_check = {glyph for glyph in (symbol, emission) if glyph}
+    if emission:
+        glyphs_to_check.update(token.strip() for token in emission.split() if token.strip())
+    if glyphs_to_check & PHASE_VOLATILE_GLYPHS:
+        score -= 3
+    if any(keyword in message for keyword in RISK_KEYWORDS):
+        score -= 1
 
     return score
 
-def minimax(commands, depth=2):
-    # Basic shallow minimax since state isn't deeply branching
+
+def select_best_command(commands: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
     best_score = float("-inf")
-    best_command = None
+    best_command: Optional[Dict[str, Any]] = None
     for cmd in commands:
         score = evaluate_command(cmd)
         if score > best_score:
             best_score = score
             best_command = cmd
     return best_command
 
-def update_log(best_command):
+
+def update_log(best_command: Dict[str, Any]) -> None:
     log = load_json(LOG_PATH)
-    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
+    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
     log_entry = {
-        "agent": best_command["agent"],
-        "symbol": best_command["symbol"],
-        "symbol_emission": best_command["symbol_emission"],
-        "message": best_command["message"],
+        "agent": best_command.get("agent", "Unknown"),
+        "symbol": best_command.get("symbol"),
+        "symbol_emission": best_command.get("symbol_emission"),
+        "message": best_command.get("message"),
         "core_compliance": best_command.get("stabilizes", False),
         "status": "Autonomous Update",
-        "timestamp": timestamp
+        "timestamp": timestamp,
     }
-    log["log"].append(log_entry)
+    log.setdefault("log", []).append(log_entry)
     write_json(LOG_PATH, log)
-    print(f"✅ Log updated with: {best_command['symbol_emission']}")
+    print(f"✅ Log updated with: {best_command.get('symbol_emission', best_command.get('symbol', '∅'))}")
+
 
-def main():
-    candidates = load_json(CANDIDATE_PATH)["candidates"]
-    best = minimax(candidates)
-    if best:
-        update_log(best)
+def main() -> None:
+    parser = argparse.ArgumentParser(description="Evaluate Wonder Engine lattice activity.")
+    parser.add_argument(
+        "--mode",
+        choices={"logs", "commands"},
+        default="logs",
+        help="Evaluation mode: analyse log history or score incoming command candidates.",
+    )
+    parser.add_argument(
+        "--apply",
+        action="store_true",
+        help="When in command mode, append the highest scoring candidate to the log.",
+    )
+    parser.add_argument(
+        "--gpt-summary",
+        action="store_true",
+        help="After analysing logs, request a GPT insight using OPENAI_API_KEY.",
+    )
+    parser.add_argument(
+        "--gpt-model",
+        default="gpt-4o-mini",
+        help="Model name to request when generating GPT insights.",
+    )
+    parser.add_argument(
+        "--gpt-max-tokens",
+        type=int,
+        default=400,
+        help="Maximum tokens to request for GPT insight generation.",
+    )
+    parser.add_argument(
+        "--gpt-provider",
+        choices={"openai", "github"},
+        default="openai",
+        help="Select the backend to fulfil GPT insights (OpenAI or GitHub Copilot).",
+    )
+    args = parser.parse_args()
+
+    if args.mode == "logs":
+        log_data = load_json(LOG_PATH)
+        entries: Iterable[Dict[str, Any]] = log_data.get("log", [])
+        summary = aggregate_log_scores(entries)
+        print_log_report(
+            summary,
+            gpt_model=args.gpt_model,
+            gpt_tokens=args.gpt_max_tokens,
+            gpt_enabled=args.gpt_summary,
+            gpt_provider=args.gpt_provider,
+        )
     else:
-        print("⚠️ No viable command found.")
+        candidates = load_json(CANDIDATE_PATH).get("candidates", [])
+        if not candidates:
+            print("⚠️ No command candidates available.")
+            return
+        best = select_best_command(candidates)
+        if best:
+            print("📡 Highest scoring command:")
+            print(json.dumps(best, indent=2, ensure_ascii=False))
+            if args.apply:
+                update_log(best)
+        else:
+            print("⚠️ No viable command found.")
+
 
 if __name__ == "__main__":
     main()
