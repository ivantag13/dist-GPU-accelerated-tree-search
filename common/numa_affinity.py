import os
from sys import exit
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

debug_mode = False

def _has_nvidia_gpu():
    """
    Checks whether the system has an NVIDIA GPU installed.
    """
    try:
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)

        if "error" in output.decode().lower():
            return False
        else:
            return True

    except Exception as e:
        if debug_mode:
            print("nvidia-smi check failed:", e)
        return False

def _has_amd_gpu():
    """
    Checks whether the system has an AMD GPU installed.
    """
    try:
        output = subprocess.check_output(['rocm-smi'], stderr=subprocess.STDOUT)

        if "error" in output.decode().lower():
            return False
        else:
            return True

    except Exception as e:
        if debug_mode:
            print("rocm-smi check failed:", e)
        return False

def _parse_numa_affinity_nvidia(filename):
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

def _parse_numa_affinity_amd(filename):
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

def _parse_cpu_affinity(filename):
    """
    Parses topology file to extract NUMA affinity for each CPU thread.
    """
    cpu_affinities = []

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        cpu_affinities.append(int(line[0]))

    return cpu_affinities

def get_numa_affinity():
    """
    Collects NUMA affinity of each CPU thread and each GPU device. The output is
    a file named "affinity.txt" in the following format:
    thread_id thread_numa_node gpu_numa_node
    """
    has_nvidia_gpu = _has_nvidia_gpu()
    has_amd_gpu = _has_amd_gpu()

    # Step 1: parses the NUMA affinity of each CPU thread and GPU device
    if has_nvidia_gpu or has_amd_gpu:
        with open("topo1.txt", "w") as f:
            subprocess.run(["lscpu", "-e=NODE"], stdout=f)
        numa_affinity_threads = _parse_cpu_affinity("topo1.txt")
        os.remove("topo1.txt")

        if has_nvidia_gpu:
            with open("topo2.txt", "w") as f:
                subprocess.run(["nvidia-smi", "topo", "-m"], stdout=f)
            numa_affinity_gpus = _parse_numa_affinity_nvidia("topo2.txt")
            os.remove("topo2.txt")

        else: #has_amd_gpu
            with open("topo2.txt", "w") as f:
                subprocess.run(["rocm-smi", "--showtoponuma"], stdout=f)
            numa_affinity_gpus = _parse_numa_affinity_amd("topo2.txt")
            os.remove("topo2.txt")

    else:
        exit("Error - No GPU found")

    # Step 2: manages cases where multiple GPUs share the same NUMA node
    special_case = False
    for i in numa_affinity_gpus:
        if numa_affinity_gpus.count(i) > 1:
            special_case = True
            break

    if special_case:
        nb_gpus = len(numa_affinity_gpus)

        threads_per_gpu = [0] * nb_gpus
        for i in range(0, nb_gpus):
            numa_node = numa_affinity_gpus[i]
            threads_per_numa = numa_affinity_threads.count(numa_node)
            gpus_per_numa = numa_affinity_gpus.count(numa_node)
            threads_per_gpu[i] = threads_per_numa / gpus_per_numa

    # Step 3: generates the output file
    with open("affinity.txt", "w") as f:
        for i in range(0, len(numa_affinity_threads)):
            thread_id = i
            thread_affinity = numa_affinity_threads[i]
            gpu_affinity = numa_affinity_gpus.index(thread_affinity)

            if special_case:
                threads_per_gpu[gpu_affinity] -= 1
                if threads_per_gpu[gpu_affinity] == 0:
                    numa_affinity_gpus[gpu_affinity] = -1

            f.write(f"{thread_id} {thread_affinity} {gpu_affinity}\n")

if __name__ == "__main__":
    get_numa_affinity()
