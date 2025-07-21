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

def get_numa_affinity():
    """
    Detects the system's GPUs (NVIDIA or AMD) and returns their NUMA affinities.
    """
    if _has_nvidia_gpu():
        with open("topo.txt", "w") as f:
            subprocess.run(["nvidia-smi", "topo", "-m"], stdout=f)
        return _get_numa_affinity_nvidia("topo.txt")

    elif _has_amd_gpu():
        with open("topo.txt", "w") as f:
            subprocess.run(["rocm-smi", "--showtoponuma"], stdout=f)
        return _get_numa_affinity_amd("topo.txt")

    else:
        exit("No GPU found...")

if __name__ == "__main__":
    print(get_numa_affinity())
