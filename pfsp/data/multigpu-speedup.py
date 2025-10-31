import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# === USER PARAMETERS ===
work_stealing_mode = 1  # 1 = enabled, 0 = disabled
exclude_instance = "30"  # instance to exclude from average plot

# === Step 1: Load CSV safely ===
df = pd.read_csv("multigpu.csv", sep=",", quotechar='"', engine="python")

# === Step 2: Parse vector columns into Python lists ===
def safe_parse_list(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x

for col in df.columns:
    if df[col].astype(str).str.startswith("[").any():
        df[col] = df[col].apply(safe_parse_list)

# === Step 3: Clean column names and numeric types ===
df.columns = [c.strip() for c in df.columns]
numeric_cols = ["nb_device", "total_tree", "total_time"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing instance_id
df = df.dropna(subset=["instance_id"])
df["instance_id"] = df["instance_id"].astype(str)

# === Step 4: Map PU counts to GPU batches ===
gpu_batches = {4: 1, 8: 2, 16: 4, 32: 8}

# === Step 5: Compute speed-ups per instance ===
records = []
for instance, group in df.groupby("instance_id"):
    # Map nb_device → total_time
    times = {gpu_batches.get(nb, nb): t for nb, t in zip(group["nb_device"], group["total_time"])}
    # Only consider rows matching chosen work_stealing_mode
    ws_flags = {gpu_batches.get(nb, nb): ws for nb, ws in zip(group["nb_device"], group["work_stealing"])}
    times = {k: v for k, v in times.items() if ws_flags[k] == work_stealing_mode}

    if 1 in times:
        base_time = times[1]
        for n_gpu, t in times.items():
            if t > 0 and not np.isnan(t):
                speedup = base_time / t
                records.append({"instance_id": instance, "n_gpu": n_gpu, "speedup": speedup})

speedup_df = pd.DataFrame(records)

# === Step 6: Order instances by total_tree ===
order = df.sort_values("total_tree")["instance_id"].dropna().unique()
order = [str(x) for x in order]  # ensure string type
speedup_df = speedup_df.dropna(subset=["instance_id"])
speedup_df["instance_id"] = speedup_df["instance_id"].astype(str)
speedup_df["instance_id"] = pd.Categorical(speedup_df["instance_id"], categories=order, ordered=True)
speedup_df.sort_values("instance_id", inplace=True)

# === Colorblind-friendly palette ===
colors = ["#0072B2", "#D55E00", "#009E73"]  # blue, reddish-orange, greenish

# === Step 7: Plot per-instance histogram with values on top ===
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.25
x = np.arange(len(order))

for i, (ng, color) in enumerate(zip([2, 4, 8], colors)):
    subset = speedup_df[speedup_df["n_gpu"] == ng]
    nc = 4 * ng
    bars = ax.bar(x + i * width, subset["speedup"], width, label=f"{ng}/{nc} GPU/CPU", color=color)

    # Add speedup labels
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width)
ax.set_xticklabels(order, rotation=45)
ax.set_xlabel("Taillard Benchmark Instances")
ax.set_ylabel("Speedup")
mode_label = "Enabled" if work_stealing_mode == 1 else "Disabled"
ax.legend(title=f"Work Stealing {mode_label}")
ax.set_title(f"MC-MGPU-B&B: Speedups per Taillard benchmark instance on LUMI (AMD MI250x)")
plt.tight_layout()
plt.savefig("intra-speedup.eps")
plt.savefig("intra-speedup.png")
plt.show()

# === Step 8: Plot average speed-up excluding 1 GPU and ta030 with values on top ===
avg_speedup = speedup_df[(speedup_df["n_gpu"] != 1) & (speedup_df["instance_id"] != exclude_instance)]
avg_speedup = avg_speedup.groupby("n_gpu")["speedup"].mean().reset_index()

print(f"Average speedups by GPU count (excluding 1 GPU and {exclude_instance}) — Work Stealing {mode_label}:")
for _, row in avg_speedup.iterrows():
    print(f"{int(row['n_gpu'])} GPUs: {row['speedup']:.2f}")

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(avg_speedup["n_gpu"].astype(str), avg_speedup["speedup"], color=colors[:len(avg_speedup)])

# Add labels on top of each bar
for bar in bars:
    height = bar.get_height()
    if not np.isnan(height):
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}", ha='center', va='bottom', fontsize=10)

ax.set_xlabel("Number of GPUs")
ax.set_ylabel("Average Speed-up")
ax.set_title(f"Average Speed-up by GPU count (excluding 1 GPU and {exclude_instance})\nWork Stealing {mode_label}")
plt.tight_layout()

plt.show()
