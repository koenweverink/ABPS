import pickle

pickle_file = "hpo_ckpts_2/ckpt_high_from0_seed101207.pkl"

# Load the pickle
with open(pickle_file, "rb") as f:
    best = pickle.load(f)

print(best.get("history", {}))