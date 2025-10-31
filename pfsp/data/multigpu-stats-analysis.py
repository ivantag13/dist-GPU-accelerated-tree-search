import pandas as pd
import numpy as np
import ast

# === CONFIG ===
filename = "multigpu.csv"   # your file name
instance_to_check = 21      # example
nb_device_to_check = 32
work_stealing_to_check = 1

pus_per_gpu = 4  # 1 GPU + 3 CPU-only PUs per group

# === STEP 1: Load CSV and parse list columns ===
def safe_parse_list(x):
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return np.nan
    return x

df = pd.read_csv(filename, sep=",", quotechar='"', engine="python")
for col in df.columns:
    if df[col].astype(str).str.startswith("[").any():
        df[col] = df[col].apply(safe_parse_list)

# === STEP 2: Convert numeric columns ===
num_cols = ["nb_device", "work_stealing", "total_time", "total_tree"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# === STEP 3: Analyze one instance (GPU timings) ===
def analyze_instance(df, instance_id, nb_device, ws):
    row = df[
        (df["instance_id"] == instance_id)
        & (df["nb_device"] == nb_device)
        & (df["work_stealing"] == ws)
    ]
    if row.empty:
        print(f"No row found for instance={instance_id}, nb_device={nb_device}, ws={ws}")
        return
    row = row.iloc[0]

    gpu_metrics = [
        "gpu_memcpy_time", "gpu_malloc_time", "gpu_kernel_time",
        "gpu_gen_child_time", "pool_ops_time", "gpu_idle_time", "termination_time"
    ]

    nb_gpus = nb_device // pus_per_gpu
    total_time = row["total_time"]

    print(f"\n=== Instance {instance_id} | nb_device={nb_device} | work_stealing={ws} ===")
    print(f"Total time: {total_time:.4f} s")
    print(f"Detected {nb_gpus} GPUs ({nb_device} total PUs)\n")

    gpu_indices = range(0, nb_device, pus_per_gpu)
    gpu_results = {m: [] for m in gpu_metrics}

    for g, idx in enumerate(gpu_indices):
        print(f"--- GPU {g} (PU index {idx}) ---")
        gpu_total = 0.0
        for metric in gpu_metrics:
            vec = row.get(metric, None)
            if not isinstance(vec, list) or len(vec) <= idx:
                print(f"{metric}: missing data")
                continue
            val = vec[idx]
            gpu_results[metric].append(val)
            gpu_total += val
        for metric in gpu_metrics:
            if len(gpu_results[metric]) == g + 1:
                perc = (gpu_results[metric][-1] / gpu_total * 100) if gpu_total > 0 else 0
                print(f"{metric:<20}: {perc:5.1f}% of GPU time")
        print(f"Total GPU {g} time: {gpu_total:.4f} s\n")

    print("=== Averages across GPUs ===")
    total_avg = np.nansum([np.nanmean(gpu_results[m]) for m in gpu_metrics])
    for metric in gpu_metrics:
        mean_val = np.nanmean(gpu_results[metric])
        mean_pct = (mean_val / total_avg * 100) if total_avg > 0 else 0
        print(f"{metric:<20} avg: {mean_val:.4f} s ({mean_pct:5.1f}%)")

def analyze_instance_workload_and_steals(df, instance_id, nb_device, ws, pus_per_gpu=4):
    sub = df[
        (df["instance_id"] == instance_id)
        & (df["nb_device"] == nb_device)
        & (df["work_stealing"] == ws)
    ]
    if sub.empty:
        print(f"No rows for instance={instance_id}, nb_device={nb_device}, ws={ws}")
        return

    row = sub.iloc[0]  # only take the first matching row

    gpu_indices = set(range(0, nb_device, pus_per_gpu))

    # --- Workload distribution ---
    trees = row.get("exp_tree_gpu", [])
    if not isinstance(trees, list) or len(trees) != nb_device:
        print("Invalid or missing workload data.")
    else:
        total = np.nansum(trees)
        if total > 0:
            gpu_sum = np.nansum([trees[i] for i in gpu_indices])
            cpu_sum = np.nansum([trees[i] for i in range(nb_device) if i not in gpu_indices])
            gpu_percent = gpu_sum / total * 100
            cpu_percent = cpu_sum / total * 100
        else:
            gpu_percent = cpu_percent = np.nan

    # --- Successful work steals ---
    steals = row.get("success_steals_gpu", [])
    if not isinstance(steals, list) or len(steals) != nb_device:
        avg_gpu_steal = avg_cpu_steal = total_steal = np.nan
    else:
        gpu_vals = [steals[i] for i in gpu_indices]
        cpu_vals = [steals[i] for i in range(nb_device) if i not in gpu_indices]
        avg_gpu_steal = np.nanmean(gpu_vals)
        avg_cpu_steal = np.nanmean(cpu_vals)
        total_steal = np.nansum(steals)

    # --- Output ---
    print(f"\n=== Instance {instance_id} | nb_device={nb_device} | work_stealing={ws} ===")
    print("\n--- Workload Distribution ---")
    print(f"GPU PUs workload: {gpu_percent:6.2f}%")
    print(f"CPU-only PUs workload: {cpu_percent:6.2f}%")

    print("\n--- Successful Steals ---")
    print(f"Average GPU PU successful steals: {avg_gpu_steal:.2f}")
    print(f"Average CPU-only PU successful steals: {avg_cpu_steal:.2f}")
    print(f"Total successful steals (sum of all PUs): {total_steal:.2f}")

# === Run analyses ===
analyze_instance(df, instance_to_check, nb_device_to_check, work_stealing_to_check)
analyze_instance_workload_and_steals(df, instance_to_check, nb_device_to_check, work_stealing_to_check)
