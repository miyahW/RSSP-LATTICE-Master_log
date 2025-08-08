import json
import time
from log_manager import write_log_entry

CANDIDATE_PATH = "Command_candidates.log"

# Glyphs recognized as stabilizers by Wonder Engine
STABILIZING_GLYPHS = {"⌘", "⌘Ω", "⧉", "∞", "⌘∞"}
PHASE_VOLATILE_GLYPHS = {"⏃", "⟠", "⊘", "⧄"}
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def evaluate_command(command):
    score = 0
    symbol = command["symbol"]
    message = command.get("message", "").lower()

    if symbol in STABILIZING_GLYPHS:
        score += 2
    if command.get("stabilizes"):
        score += 2
    if any(keyword in message for keyword in ["stabilize", "anchor", "contain"]):
        score += 1
    if symbol in PHASE_VOLATILE_GLYPHS:
        score -= 3
    if "recurse" in message:
        score -= 1  # unanchored recursion risk
    if "diverge" in message:
        score -= 2  # harder penalty for known fracturing
    if "echo" in message:
        score -= 1  # echo = unstable without mirror anchor

    return score

def minimax(commands, depth=2):
    # Basic shallow minimax since state isn't deeply branching
    best_score = float("-inf")
    best_command = None
    for cmd in commands:
        score = evaluate_command(cmd)
        if score > best_score:
            best_score = score
            best_command = cmd
    return best_command

def update_log(best_command):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    log_entry = {
        "agent": best_command["agent"],
        "symbol": best_command["symbol"],
        "symbol_emission": best_command["symbol_emission"],
        "message": best_command["message"],
        "core_compliance": best_command.get("stabilizes", False),
        "status": "Autonomous Update",
        "timestamp": timestamp
    }
    write_log_entry(log_entry)
    print(f"✅ Log updated with: {best_command['symbol_emission']}")

def main():
    candidates = load_json(CANDIDATE_PATH)["candidates"]
    best = minimax(candidates)
    if best:
        update_log(best)
    else:
        print("⚠️ No viable command found.")

if __name__ == "__main__":
    main()
