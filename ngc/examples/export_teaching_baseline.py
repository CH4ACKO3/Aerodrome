"""Regenerate the online lesson's reference from the actual World simulation."""
import json
from pathlib import Path
from aerodrome.teaching.experiment import run_velocity

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[2]/"website/public/examples/rigid-velocity-v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = run_velocity({})
    result["source"] = "precomputed"
    target.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(target)
