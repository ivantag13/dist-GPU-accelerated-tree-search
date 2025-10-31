import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set(style="whitegrid", palette="colorblind")

# === USER PARAMETERS ===
intra_file = "multigpu.csv"
dist_file = "dist_multigpu.csv"
nb_device = 32
comm_sizes = [2, 4, 8, 16, 32, 64, 128]
compare_lbs = [0, 2]  # no LB vs WS
instances_to_plot = ["24", "21"]  # Only these two for comparison

# Chapel average runtimes (per instance)
chapel_times = {
    "24": [1866.64333333333, 1086.43, 541.011333333333, 273.557333333333, 136.939333333333,
           69.3992333333333, 36.0103666666667, 20.3627333333333],
    "21": [6600.81333333333, 3941.09666666667, 1967.77, 984.748666666667, 495.009,
           247.914666666667, 125.867333333333, 67.1628666666667]
}
chapel_comm_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

# --- Step A: Load intra-node reference ---
intra_df = pd.read_csv(intra_file, engine="python")
intra_df.columns = [c.strip() for c in intra_df.columns]
intra_df['instance_id'] = intra_df['instance_id'].astype(str)

# ✅ Keep only the right configuration
ref_df = intra_df[(intra_df["nb_device"] == nb_device) & (intra_df["work_stealing"] == 1)]
if ref_df.empty:
    raise ValueError(f"No intra-node runs found for nb_device={nb_device} and WS=1")

# Compute per-instance reference times
ref_times = ref_df.groupby("instance_id")["total_time"].mean().to_dict()

print("Reference intra-node times (avg over WS runs, nb_device == 32):")
for k, v in sorted(ref_times.items()):
    print(f"  Instance {k}: {v:.4f}")


# --- Step B: Load distributed CSV ---
dist_df = pd.read_csv(dist_file, engine="python")
dist_df.columns = [c.strip() for c in dist_df.columns]
dist_df['instance_id'] = dist_df['instance_id'].astype(str)

def safe_parse_list(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x

vector_cols = [c for c in dist_df.columns if c.startswith("all_")]
for col in vector_cols:
    dist_df[col] = dist_df[col].apply(safe_parse_list)

# --- Step C: Build cleaned distributed averages ---
records = []
for inst in instances_to_plot:
    inst_df = dist_df[dist_df['instance_id'] == inst]
    for cs in comm_sizes:
        for lb in compare_lbs:
            subset = inst_df[(inst_df['comm_size'] == cs) & (inst_df['load_balancing'] == lb)]
            valid = subset[subset['all_exp_tree_gpu'].apply(lambda v: isinstance(v, list) and len(v) == nb_device * cs)]
            if valid.empty:
                continue
            avg_time = valid['total_time'].mean()
            records.append({
                'instance_id': inst,
                'comm_size': cs,
                'load_balancing': lb,
                'total_time': avg_time
            })

dist_clean_df = pd.DataFrame(records)

# --- Step D: Compute speedups for your implementation ---
def compute_speedup(row):
    inst = row['instance_id']
    ref = ref_times.get(inst, np.nan)
    if np.isnan(ref) or row['total_time'] == 0:
        return np.nan
    return ref / row['total_time']

dist_clean_df['speedup'] = dist_clean_df.apply(compute_speedup, axis=1)

# --- Step E: Compute Chapel speedups ---
chapel_speedups_abs = {}
chapel_speedups_rel = {}

for inst in instances_to_plot:
    times = chapel_times[inst]
    ref_yours = ref_times.get(inst, np.nan)
    ref_chapel = times[0]  # its intra-node value

    abs_speeds = [ref_yours / t for t in times]
    rel_speeds = [ref_chapel / t for t in times]

    chapel_speedups_abs[inst] = abs_speeds
    chapel_speedups_rel[inst] = rel_speeds

# --- Step F: Plot ---
for inst in instances_to_plot:
    plt.figure(figsize=(8, 6))

    # Your speedups (LB=2)
    df_i = dist_clean_df[(dist_clean_df["instance_id"] == inst) & (dist_clean_df["load_balancing"] == 2)]
    plt.plot(df_i["comm_size"], df_i["speedup"], marker="o", label=f"MN-MC-MGPU-B&B DWS")

    # Chapel absolute and relative
    plt.plot(chapel_comm_sizes, chapel_speedups_abs[inst], marker="s", linestyle="--", label="ChplBB Absolute Speedup")
    plt.plot(chapel_comm_sizes, chapel_speedups_rel[inst], marker="^", linestyle=":", label="ChplBB Relative Speedup")

    plt.plot([1, max(chapel_comm_sizes)], [1, max(chapel_comm_sizes)], "k--", lw=0.8, label="Linear Speedup")

    plt.xlabel("Compute Nodes")
    plt.ylabel("Speedup")
    plt.title(f"MN-MC-MGPU-B&B vs. ChplBB: Speedup Comparison for Instance ta0{inst}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"speedup_comparison_instance_{inst}.png")
    plt.savefig(f"speedup_comparison_instance_{inst}.eps")
    plt.show()
