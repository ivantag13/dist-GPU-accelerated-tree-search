import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast

# === USER PARAMETERS ===
filename = "multigpu.csv"
pus_per_gpu = 4  # 1 GPU + 3 CPU-only PUs
max_devices = None  # You can override if needed, e.g., max_devices = 32

# === STEP 1: Load CSV safely ===
df = pd.read_csv(filename, sep=",", quotechar='"', engine="python")

# === STEP 2: Parse list-like vector columns ===
def safe_parse_list(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return np.nan
    return x

for col in df.columns:
    if df[col].astype(str).str.startswith("[").any():
        df[col] = df[col].apply(safe_parse_list)

# === STEP 3: Clean and convert ===
df.columns = [c.strip() for c in df.columns]
numeric_cols = ["nb_device", "total_tree", "total_time", "work_stealing"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["instance_id", "nb_device"])

# === STEP 4: Detect max GPU group ===
if max_devices is None:
    max_devices = df["nb_device"].max()

max_gpus = max_devices // pus_per_gpu
print(f"Detected {max_gpus} GPUs (total PUs = {max_devices})")

# === STEP 5: Filter dataset for max nb_devices only ===
df_max = df[df["nb_device"] == max_devices].copy()

# === STEP 6: Select best (lowest total_time) repetition per (instance_id, work_stealing) ===
group_cols = ["instance_id", "work_stealing"]
idx = df_max.dropna(subset=["total_time"]).groupby(group_cols)["total_time"].idxmin()
df_best = df_max.loc[idx].reset_index(drop=True)

# === STEP 7: Extract explored-tree info per GPU group ===
rows = []
for _, row in df_best.iterrows():
    instance = str(row["instance_id"])
    ws = "Enabled" if row["work_stealing"] == 1 else "Disabled"
    total_tree = row["total_tree"]

    exp_tree_gpu = row.get("exp_tree_gpu", None)
    if not isinstance(exp_tree_gpu, list):
        continue

    # Number of GPU groups
    gpu_groups = len(exp_tree_gpu) // pus_per_gpu

    # Sum work done by each GPU group (GPU PU + its CPU PUs)
    for g in range(gpu_groups):
        start = g * pus_per_gpu
        end = start + pus_per_gpu
        group_sum = np.nansum(exp_tree_gpu[start:end])
        workload_percent = (group_sum / total_tree) * 100 if total_tree > 0 else np.nan

        rows.append({
            "Instance": instance,
            "WS": ws,
            "GPU_Group": g,
            "Workload (%)": workload_percent,
            "total_tree": total_tree
        })

df_plot = pd.DataFrame(rows)

# === STEP 8: Order instances by total_tree ===
order = (
    df_plot.groupby("Instance")["total_tree"]
    .mean()
    .sort_values()
    .index
)

df_plot["Instance"] = pd.Categorical(df_plot["Instance"], categories=order, ordered=True)
df_plot = df_plot.sort_values("Instance")

# === STEP 9: Plot color-blind-friendly boxplot with WS Disabled on left ===
fs = 15
plt.figure(figsize=(16, 6))
sns.set_palette("colorblind")  # color-blind friendly palette

sns.boxplot(
    x="Instance",
    y="Workload (%)",
    hue="WS",
    data=df_plot,
    order=order,
    hue_order=["Disabled", "Enabled"]  # ✅ ensures Disabled is on the left
)

plt.title(f"MC-MGPU-B&B: Workload Balance per GPU Group Across Benchmark Instances ({max_gpus}/32 GPU/CPU)", fontsize=fs)
plt.xlabel("Taillard Benchmark Instances", fontsize=fs)
plt.ylabel("Workload (%)", fontsize=fs)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.grid(axis="y", linestyle='--', linewidth=0.5)
plt.legend(title="Work Stealing", fontsize=fs, title_fontsize=fs)
plt.tight_layout()
plt.savefig("boxplot-workstealing.png", dpi=300, bbox_inches='tight')
plt.savefig("boxplot-workstealing.eps", bbox_inches='tight')
plt.show()
