import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="colorblind")

# === USER PARAMETERS ===
filename = "dist_multigpu.csv"
target_instances = ["26", "28", "23", "24", "21"]
lb_target = 2

# --- Step 1: Load file ---
df = pd.read_csv(filename, engine="python")
df.columns = [c.strip() for c in df.columns]
df["instance_id"] = df["instance_id"].astype(str)

# --- Step 2: Parse vector column ---
def safe_parse_list(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x

df["all_dist_load_bal"] = df["all_dist_load_bal"].apply(safe_parse_list)

# --- Step 3: Select valid rows (SECOND valid row logic) ---
records = []
for inst in target_instances:
    df_inst = df[(df["instance_id"] == inst) & (df["load_balancing"] == lb_target)]
    for cs in sorted(df_inst["comm_size"].unique()):
        subset = df_inst[df_inst["comm_size"] == cs]

        # Collect all valid rows for this (instance, comm_size)
        valid_rows = []
        for _, r in subset.iterrows():
            lst = r["all_dist_load_bal"]
            if isinstance(lst, list) and len(lst) == cs:
                valid_rows.append(r)

        # Take the second valid row if available; otherwise, the first one
        if len(valid_rows) == 0:
            continue
        elif len(valid_rows) == 1:
            row = valid_rows[0]
        else:
            row = valid_rows[1]

        # Compute metrics
        lst = row["all_dist_load_bal"]
        avg_ws_per_proc = np.mean(lst)
        total_ws = sum(lst)
        total_time = row["total_time"]
        avg_ws_per_sec = avg_ws_per_proc / total_time if total_time > 0 else np.nan

        records.append({
            "instance_id": inst,
            "comm_size": cs,
            "avg_work_stealing_per_proc": avg_ws_per_proc,
            "avg_ws_per_sec": avg_ws_per_sec,
            "total_work_stealing": total_ws,
            "total_time": total_time
        })

res_df = pd.DataFrame(records)

# --- Step 4: Plot total work stealing vs comm_size ---
plt.figure(figsize=(8, 6))
for inst in target_instances:
    df_i = res_df[res_df["instance_id"] == inst]
    plt.plot(df_i["comm_size"], df_i["total_work_stealing"], marker="o", label=f"Instance {inst}")
plt.xlabel("MPI Processes")
plt.ylabel("Total Number of DWS")
plt.title("Evolution of DWS Against Number of Compute Nodes")
plt.legend()
plt.tight_layout()
plt.savefig("dist_workstealing_total.png")
plt.savefig("dist_workstealing_total.eps")
plt.show()

# --- Step 5: Plot average work stealing per process per second ---
plt.figure(figsize=(8, 6))
for inst in target_instances:
    df_i = res_df[res_df["instance_id"] == inst]
    plt.plot(df_i["comm_size"], df_i["avg_ws_per_sec"], marker="s", label=f"Instance {inst}")
plt.xlabel("Compute Nodes (MPI Processes)")
plt.ylabel("Average DWS per Process per Second")
plt.title("MN-MC-MGPU-B&B: Evolution of Average DWS Against Number of Compute Nodes")
plt.legend()
plt.tight_layout()
plt.savefig("dist_workstealing_avg_rate.png")
plt.savefig("dist_workstealing_avg_rate.eps")
plt.show()
