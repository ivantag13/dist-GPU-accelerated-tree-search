import os
from sys import exit
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def _has_nvidia_gpu():
    """
    Checks whether the system has an NVIDIA GPU installed.
    """
    try:
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def _has_amd_gpu():
    """
    Checks whether the system has an AMD GPU installed.
    """
    try:
        output = subprocess.check_output(['rocm-smi'], stderr=subprocess.STDOUT)
        return True
    except Exception:
        return False

def _get_numa_affinity_nvidia(filename):
    """
    Parses topology file to extract NUMA affinity for each Nvidia GPU.
    """
    matrix = []
    with open(filename, 'r') as f:
        for line in f:
            row = line.split('\t')
            matrix.append(row)

    numa_affinities = []

    for row in matrix:
        if row[0].startswith('GPU'):
            numa_affinity = row[-3]
            numa_affinities.append(int(numa_affinity))

    return numa_affinities

def _get_numa_affinity_amd(filename):
    """
    Parses topology file to extract NUMA affinity for each AMD GPU.
    """
    numa_affinities = []

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if "Numa Affinity:" in line:
            parts = line.split(":")
            numa_id = int(parts[-1].strip())
            numa_affinities.append(numa_id)

    return numa_affinities

def _get_cpu_affinity(filename):
    cpu_affinities = []

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        cpu_affinities.append(int(line[0]))

    return cpu_affinities

def get_numa_affinity():
    """
    Detects the system's GPUs (NVIDIA or AMD) and returns their NUMA affinities.
    """
    has_nvidia_gpu = _has_nvidia_gpu()
    has_amd_gpu = _has_amd_gpu()

    # Step 1: parses the NUMA/CPU affinities
    if has_nvidia_gpu or has_amd_gpu:
        # NOTE: there are false-positives when rocm-smi or nvidia-smi is installed
        # but no GPUs are available (e.g., LUMI login nodes)
        with open("topo1.txt", "w") as f:
            subprocess.run(["lscpu", "-e=NODE"], stdout=f)
        cpu_affinity = _get_cpu_affinity("topo1.txt")
        os.remove("topo1.txt")

        if has_nvidia_gpu:
            with open("topo2.txt", "w") as f:
                subprocess.run(["nvidia-smi", "topo", "-m"], stdout=f)
            numa_affinity = _get_numa_affinity_nvidia("topo2.txt")
            os.remove("topo2.txt")

        else: #has_amd_gpu
            with open("topo2.txt", "w") as f:
                subprocess.run(["rocm-smi", "--showtoponuma"], stdout=f)
            numa_affinity = _get_numa_affinity_amd("topo2.txt")
            os.remove("topo2.txt")

    else:
        exit("Error - No GPU found")

    # Step 2: generates the output file
    with open("affinity.txt", "w") as f:
        for i in range(0, len(cpu_affinity)):
            thread_id = i
            thread_affinity = cpu_affinity[i]
            # NOTE: there are edge-cases where two GPUs share the same NUMA node
            # (e.g., LUMI).
            gpu_affinity = numa_affinity.index(thread_affinity)
            f.write(f"{thread_id} {thread_affinity} {gpu_affinity}\n")

if __name__ == "__main__":
    get_numa_affinity()
