# inspect_best_artifact.py
import os, json, pickle, glob, types, collections
from pprint import pprint

HPO_JSON = "hpo_minimal_results.json"  # adjust if needed

# 1) Get best_path from HPO results, or discover one
best_path = None
if os.path.exists(HPO_JSON):
    with open(HPO_JSON, "r", encoding="utf-8") as f:
        hpo = json.load(f)
    final_pick = hpo.get("final_pick", {}) or {}
    # common locations
    best_path = (final_pick.get("best_path")
                 or (final_pick.get("cfg_used") or {}).get("best_path"))
if not best_path:
    # discover best_*.pkl in current tree (fallback)
    candidates = glob.glob("**/best_*.pkl", recursive=True)
    best_path = sorted(candidates, key=len)[0] if candidates else None

if not best_path or not os.path.exists(best_path):
    raise FileNotFoundError(f"Could not locate a best_*.pkl. Looked for: {best_path or '(none)'}")

print(f"[i] Loading best artifact: {best_path}")

# 2) Load pickle (may be a custom class, tuple, or dict)
with open(best_path, "rb") as f:
    obj = pickle.load(f)

def to_mapping(x):
    """Return a dict-like view for objects when possible."""
    if isinstance(x, dict):
        return x
    if hasattr(x, "__dict__") and isinstance(x.__dict__, dict):
        return x.__dict__
    if isinstance(x, (list, tuple)) and len(x) == 2 and isinstance(x[0], str):
        # sometimes ('sequence', {...}) or ('chromosome', {...})
        return dict([x])  # crude; best-effort
    return None

# 3) Recursive search for candidate keys
CONFIG_KEYS = {"config", "cfg", "friendly_config", "settings"}
SEQ_KEYS    = {"sequence", "attack_sequence", "plan", "actions", "genes"}

def walk_find(root, want_keys):
    """Yield (path, value) for any mapping key that matches want_keys."""
    seen = set()
    def _walk(x, path="root"):
        if id(x) in seen:
            return
        seen.add(id(x))
        m = to_mapping(x)
        if m is not None:
            for k, v in m.items():
                if k in want_keys:
                    yield (f"{path}.{k}", v)
                yield from _walk(v, f"{path}.{k}")
        elif isinstance(x, (list, tuple, set)):
            for i, v in enumerate(x):
                yield from _walk(v, f"{path}[{i}]")
    yield from _walk(root)

config_hits = list(walk_find(obj, CONFIG_KEYS))
seq_hits    = list(walk_find(obj, SEQ_KEYS))

print("\n[i] Candidate CONFIG fields found:")
for p, _ in config_hits[:10]:
    print("  -", p)
if len(config_hits) > 10:
    print("  ...")

print("\n[i] Candidate SEQUENCE fields found:")
for p, _ in seq_hits[:10]:
    print("  -", p)
if len(seq_hits) > 10:
    print("  ...")

# 4) Extract “best guess” values
def first_value(hits):
    return hits[0][1] if hits else None

config = first_value(config_hits)
sequence = first_value(seq_hits)

# 5) Fallback: try top-level variants and show keys/types
def top_keys(x):
    m = to_mapping(x)
    return list(m.keys()) if m else []

if config is None or sequence is None:
    print("\n[!] Could not confidently extract both fields.")
    print("[i] Top-level keys/types to help you identify the right names:")
    m = to_mapping(obj)
    if m:
        for k, v in m.items():
            print(f"  - {k}: {type(v).__name__}")
    else:
        print(f"  (object type: {type(obj).__name__})")

# 6) Last fallback for config: use cfg_used from HPO JSON if available
if (config is None) and "hpo" in locals():
    cfg_used = (final_pick.get("cfg_used") or {})
    if cfg_used:
        print("\n[i] Falling back to cfg_used from HPO results for CONFIG.")
        config = cfg_used

# 7) Print concise previews
print("\n=== PREVIEW ===")
print("CONFIG present:", config is not None)
print("SEQUENCE present:", sequence is not None)
if isinstance(sequence, (list, tuple)):
    print(f"SEQUENCE length: {len(sequence)}")
    print("First item preview:", sequence[0] if sequence else None)
elif sequence is not None:
    print("SEQUENCE type:", type(sequence).__name__)

# Optional: dump to JSON if both are JSON-serializable
try:
    import json
    out = {"config": config, "sequence": sequence if isinstance(sequence, (list, tuple, dict)) else str(type(sequence))}
    with open("best_artifact_extracted_2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n[i] Wrote best_artifact_extracted_2.json")
except Exception as e:
    print("\n[!] Could not write extraction JSON:", e)
