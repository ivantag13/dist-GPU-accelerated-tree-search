import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast

sns.set_palette("colorblind")  # color-blind friendly palette

# === USER PARAMETERS ===
intra_file = "multigpu.csv"
dist_file = "dist_multigpu.csv"
nb_device = 32
comm_sizes = [2,4,8,16,32,64,128]
compare_lbs = [0, 2]  # no LB vs work stealing
top_n_instances = 3

# --- Step A: Load intra-node reference ---
intra_df = pd.read_csv(intra_file, engine="python")
intra_df.columns = [c.strip() for c in intra_df.columns]
intra_df['instance_id'] = intra_df['instance_id'].astype(str)

# Pick max nb_device
max_nb = int(intra_df['nb_device'].max())
ref_df = intra_df[(intra_df['work_stealing'] == 1) & (intra_df['nb_device'] == max_nb)]
ref_times = ref_df.groupby('instance_id')['total_time'].mean().to_dict()
print("Reference intra-node times (avg over WS runs):")
for k,v in sorted(ref_times.items()):
    print(f"  Instance {k}: {v:.4f}")

# --- Step B: Load distributed CSV ---
dist_df = pd.read_csv(dist_file, engine="python")
dist_df.columns = [c.strip() for c in dist_df.columns]
dist_df['instance_id'] = dist_df['instance_id'].astype(str)

def safe_parse_list(x):
    if isinstance(x,str) and x.startswith("[") and x.endswith("]"):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return x

all_vector_cols = [c for c in dist_df.columns if c.startswith("all_")]
for col in all_vector_cols:
    dist_df[col] = dist_df[col].apply(safe_parse_list)

# --- Step C: Select top N instances by total_tree ---
#largest_instances = dist_df.groupby('instance_id')['total_tree'].max().nlargest(top_n_instances).index.tolist()
#print("\nTop instances selected (largest total_tree):", largest_instances)
# --- Step C: Select top N instances by total_tree, ascending order ---
top_instances = dist_df.groupby('instance_id')['total_tree'].max()
largest_instances = top_instances.nlargest(top_n_instances).sort_values(ascending=True).index.tolist()
print("\nTop instances selected (smallest to largest total_tree):", largest_instances)


# --- Step D: Build cleaned distributed averages ---
records = []
for inst in largest_instances:
    inst_df = dist_df[dist_df['instance_id'] == inst]
    for cs in comm_sizes:
        for lb in compare_lbs:
            subset = inst_df[(inst_df['comm_size']==cs) & (inst_df['load_balancing']==lb)]
            # Valid vector lengths
            valid = subset[subset['all_exp_tree_gpu'].apply(lambda v: isinstance(v,list) and len(v) == nb_device*cs)]
            if valid.empty:
                continue
            avg_time = valid['total_time'].mean()
            avg_vector = np.mean(np.vstack(valid['all_exp_tree_gpu'].to_list()), axis=0).tolist()
            records.append({
                'instance_id': inst,
                'comm_size': cs,
                'load_balancing': lb,
                'total_time': avg_time,
                'all_exp_tree_gpu': avg_vector
            })

dist_clean_df = pd.DataFrame(records)

# --- Print distributed averages used for speed-ups ---
print("\nDistributed averages used for speed-ups:")
for inst in largest_instances:
    print(f"\nInstance {inst}:")
    for cs in comm_sizes:
        for lb in compare_lbs:
            row = dist_clean_df[(dist_clean_df['instance_id']==inst) &
                                (dist_clean_df['comm_size']==cs) &
                                (dist_clean_df['load_balancing']==lb)]
            if not row.empty:
                print(f"  comm_size={cs}, LB={lb}: avg total_time={row['total_time'].values[0]:.4f}")

# --- Step E: Compute speed-ups ---
def compute_speedup(row):
    inst = row['instance_id']
    ref = ref_times.get(inst,np.nan)
    if np.isnan(ref) or row['total_time']==0:
        return np.nan
    return ref/row['total_time']

dist_clean_df['speedup'] = dist_clean_df.apply(compute_speedup, axis=1)

# --- Step F: Plot speed-ups ---
fig, axes = plt.subplots(len(largest_instances), 1, figsize=(8, 5 * len(largest_instances)), squeeze=False)
axes = axes.flatten()  # make it a simple 1D list of axes

for ax, inst in zip(axes, largest_instances):
    df_inst = dist_clean_df[dist_clean_df['instance_id']==inst]
    x = np.arange(len(comm_sizes))
    width = 0.35

    speeds_lb0 = [df_inst[(df_inst['comm_size']==cs) & (df_inst['load_balancing']==0)]['speedup'].values[0]
                  if not df_inst[(df_inst['comm_size']==cs) & (df_inst['load_balancing']==0)].empty else np.nan
                  for cs in comm_sizes]
    speeds_lb2 = [df_inst[(df_inst['comm_size']==cs) & (df_inst['load_balancing']==2)]['speedup'].values[0]
                  if not df_inst[(df_inst['comm_size']==cs) & (df_inst['load_balancing']==2)].empty else np.nan
                  for cs in comm_sizes]

    bars0 = ax.bar(x - width/2, speeds_lb0, width)
    bars2 = ax.bar(x + width/2, speeds_lb2, width)

    all_speeds = [s for s in speeds_lb0 + speeds_lb2 if not np.isnan(s)]
    ylim = (0, max(all_speeds)*1.15 if all_speeds else 1)
    ax.set_ylim(ylim)

    # Labels on top
    for bar in bars0 + bars2:
        h = bar.get_height()
        if np.isnan(h):
            continue
        ax.text(bar.get_x()+bar.get_width()/2, h + 0.02*max(1,h), f"{h:.1f}", ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in comm_sizes], rotation=45)
    ax.set_xlabel("Compute Nodes (MPI processes)")
    ax.set_ylabel("Speedup")
    ax.set_title(f"Instance ta0{inst}")
    ax.legend([bars0[0], bars2[0]], ["None", "DWS"])

plt.suptitle("MN-MC-MGPU-B&B Speedups Comparison:\nNo inter-node load balancing (None) vs. Inter-node work stealing (DWS)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("speedup-dist.eps")
plt.savefig("speedup-dist.png")
plt.show()



# ---------------- Step G: compute workload percentages ----------------
workload_records = []
for inst in largest_instances:
    inst_df = dist_clean_df = dist_df[dist_df['instance_id']==inst]
    for cs in comm_sizes:
        for lb in compare_lbs:
            row = inst_df[(inst_df['comm_size']==cs) & (inst_df['load_balancing']==lb)]
            if row.empty:
                continue
            row = row.iloc[0]
            tree_vector = row['all_exp_tree_gpu']
            if not isinstance(tree_vector,list) or len(tree_vector)!=nb_device*cs:
                continue
            workload = [sum(tree_vector[i:i+nb_device]) for i in range(0,len(tree_vector),nb_device)]
            total = sum(workload)
            workload_percent = [w/total*100 for w in workload]
            for j,val in enumerate(workload_percent):
                workload_records.append({'instance_id': inst,
                                         'comm_size': cs,
                                         'load_balancing': lb,
                                         'process': j,
                                         'workload_percent': val})
workload_df = pd.DataFrame(workload_records)

# ---------------- Step H: plot workload boxplots ----------------
for inst in largest_instances:
    fig, ax = plt.subplots(figsize=(10,6))
    subset = workload_df[workload_df['instance_id']==inst]
    # prepare box data: list of lists for LB0 and LB2 per comm_size
    box_data = []
    positions = []
    pos = 1
    width = 0.3
    for cs in comm_sizes:
        for i, lb in enumerate(compare_lbs):
            vals = subset[(subset['comm_size']==cs) & (subset['load_balancing']==lb)]['workload_percent'].values
            if len(vals)==0:
                continue
            box_data.append(vals)
            positions.append(pos + i*width)
        pos += 1
    bplot = ax.boxplot(box_data, positions=positions, widths=width, patch_artist=True)
    for patch, i in zip(bplot['boxes'], range(len(box_data))):
        patch.set_facecolor(colors[i%2])
    # set x-ticks at center of LB0/LB2 pair
    xticks = [(positions[i]+positions[i+1])/2 for i in range(0,len(positions),2)]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(cs) for cs in comm_sizes])
    ax.set_xlabel("Compute Nodes (MPI processes)", fontsize=13)
    ax.set_ylabel("Workload per MPI process (%)", fontsize=13)
    ax.set_title(f"MN-MC-MGPU-B&B Workload Distribution - Instance ta0{inst}")
    ax.legend([bplot['boxes'][0], bplot['boxes'][1]], ["None", "DWS"])
    plt.tight_layout()
    plt.savefig(f"workload-dist-{inst}.eps")
    plt.savefig(f"workload-dist-{inst}.png")
    plt.show()
