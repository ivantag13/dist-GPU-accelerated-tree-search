[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15828954.svg)](https://doi.org/10.5281/zenodo.15828954)

# Distributed GPU-accelerated tree search

This repository contains the implementation of a GPU-accelerated Branch-and-Bound algorithm in C language.
The GPU-acceleration is CUDA-based, while the intra-node memory is managed with OpenMP. All inter-node communication is managed by MPI APIs.
The works of [1,2,3] instantiate on the backtracking method to solve instances of the N-Queens problem (proof-of-concept) and on the Branch-and-Bound method to solve Taillard's instances of the Permutation Flowshop Scheduling Problem (PFSP).

## Design

The algorithm is based on a general multi-pool approach, where each CPU manages its own pool of work.
We assume that one GPU is assigned per CPU core in a multi-threaded implementation (OpenMP+CUDA or MPI+OpenMP+CUDA).
Intra-node multi-GPU implementation counts with a dynamic load balancing mechanism based on an atomic spin lock system.
The tree exploration starts on the CPU, and each node taken from the work pool is evaluated, potentially pruned, and branched.
In order to exploit GPU-acceleration, we offload a chunk of nodes on the GPU when the pool size is sufficiently large.
When the GPU retrieves the nodes, the latter are evaluated in parallel and the results are sent back to the CPU, which uses them to prune or branch the nodes.
This process is repeated until the pool is empty.

## Implementations

The [nqueens](./nqueens/) and [pfsp](./pfsp/) directories contains following implementations:
- `[nqueens/pfsp]_c.c`: sequential version (C);
- `[nqueens/pfsp]_gpu_cuda.c`: single-GPU version (C+CUDA);
- `[nqueens/pfsp]_multigpu_cuda.c`: multi-GPU version (C+OpenMP+CUDA).
- `pfsp_dist_multigpu_cuda.c`: distributed inter-node multi-GPU version (C+MPI+OpenMP+CUDA).

In order to compile and execute the CUDA-based code on AMD GPU architectures, we use the `hipify-perl` tool which translates it into portable HIP C++ automatically.

## Getting started

### Setting the environment configuration

By default, in our makefile the target GPU architectures are set for two cluster environements, [Grid'5000](https://www.grid5000.fr/) and [LUMI](https://www.lumi-supercomputer.eu/).

### Compilation

All the code is compiled using the provided makefile. When doing **`make`** the user can choose between three options. The first one is by adding no additional flags, which is convenient for a local compilation. Or by adding the flag **`SYSTEM`**. For instance:
- **`make SYSTEM=g5k`**: set the compilation for the Grid'5000 system (AMD GPU architecture code set to `gfx906`)
- **`make SYSTEM=lumi`**: set the compilation for the LUMI system (AMD GPU architecture code set to `gfx90a`)
  - a bash file is available to load all concerning modules, run on terminal from this project root directory **`source config/lumi-module-load.sh`**

The CUDA architecture code is always set to `sm_80`.

### Execution

Common command-line options:
- **`-m`**: minimum number of elements to offload on a GPU device
  - any positive integer (`25` by default)

- **`-M`**: maximum number of elements to offload on a GPU device
  - any positive integer greater than `--m` (`50,000` by default)

- **`-D`**: number of GPU device(s) (only in multi-GPU setting)
  - any positive integer, typically the number of GPU devices (`1` by default)

Problem-specific command-line options:
- N-Queens:
  - **`-N`**: number of queens
    - any positive integer (`14` by default)

  - **`-g`**: number of safety check(s) per evaluation
    - any positive integer (`1` by default)

- PFSP:
  - **`-i`**: Taillard's instance to solve
    - any positive integer between `001` and `120` (`014` by default)

  - **`-w`**: Intra-node Work Stealing in `pfsp_multigpu_cuda.c`
    - `0`: disable intra-node work stealing
    - `1`: enable intra-node work stealing (default)

  - **`-L`**: Inter-node Dynamic Load Balancing in `pfsp_dist_multigpu_cuda.c`
    - `0`: no inter-node load balancing (default)
    - `1`: enable inter-node work sharing
    - `2`: enable inter-node work stealing


  <!-- TODO: give references -->
  - **`-l`**: lower bound function
    - `1`: one-machine bound which can be computed in $\mathcal{O}(mn)$ steps per subproblem (default), a.k.a., `LB1`
    - `0`: fast implementation of `lb1`, which can be compute in $\mathcal{O}(m)$ steps per subproblem, a.k.a., `LB1_d`
    - `2`: two-machine bound which can be computed in $\mathcal{O}(m^2n)$ steps per subproblem, , a.k.a., `LB2`
    <!-- a two-machine bound which relies on the exact resolution of two-machine problems obtained by relaxing capacity constraints on all machines, with the exception of a pair of machines \(M<sub>u</sub>,M<sub>v</sub>\)<sub>1<=u<v<=m</sub>, and taking the maximum over all $\frac{m(m-1)}{2}$ machine-pairs. It can be computed in $\mathcal{O}(m^2n)$ steps per subproblem. -->

  - **`-u`**: initial upper bound (UB)
    - `0`: initialize the UB to $+\infty$, leading to a search from scratch
    - `1`: initialize the UB to the best solution known (default)

Unstable command-line options:
- **`-p`**: percentage of the total size of the victim's pool to steal in WS (only in CUDA-based multi-GPU implementation)
  - any real number between `0.0` and `1.0` (`0.5` by default)

### Examples

- CUDA multi-GPU launch to solve the `ta029` Taillard instance using 4 GPU devices with `m` equal to 30 and `M` equal to 10000 and no Work Stealing enable:
```
./pfsp_multigpu_cuda.out -i 17 -D 4 -m 30 -M 10000 -w 0
```
- CUDA distributed multi-GPU launch to solve the `ta021` Taillard instance using 3 MPI processes, 8 GPUs, and LB2 bounding function on Grid'5000:
```
mpirun --hostfile hostfile --map-by ppr:1:node  --mca pml ucx --mca btl ^openib -np 3 ./pfsp_dist_multigpu_cuda.out -i 21 -D 8 -l 2
```
- HIP distributed multi-GPU launch to solve the `ta024` Taillard instance using 5 MPI processes, 7 GPUs, and LB2 bounding function on LUMI:
```
srun --nodes=5 --gpus=7 --cpus-per-task=32 --ntasks-per-node=1 ./pfsp_dist_multigpu_hip.out -i 24 -l 2 -D 7
```

## Publications

1. I. Tagliaferro, G. Helbecque, E. Krishnasamy, N. Melab, and G. Danoy. A Portable Branch-and-Bound Algorithm for Cross-Architecture Multi-GPU Systems. In 2025 HeteroPar Workshop3 from the Euro-Par4 conference, Dresden, Germany, August 26, 2025. (Accepted)
2. I. Tagliaferro, G. Helbecque, E. Krishnasamy, N. Melab, and G. Danoy. Performance and Portability in Multi-GPU Branch-and-Bound: Chapel Versus CUDA and HIP for Tree-Based Optimization. In 2025 IEEE International Parallel and Distributed Processing Symposium2 Workshops (IPDPSW), Milan, Italy, June 03–07, 2025, Proceedings, pages 1291–1293, 2025. (Presented and waiting for publication).

## Bibliography

1. G. Helbecque, E. Krishnasamy, T. Carneiro, N. Melab, and P. Bouvry. A Chapel-Based Multi-GPU Branch-and-Bound Algorithm. *Euro-Par 2024: Parallel Processing Workshops*, Madrid, Spain, 2025, pp. 463-474. DOI: [10.1007/978-3-031-90200-0_37](https://doi.org/10.1007/978-3-031-90200-0_37).
2. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU-Accelerated Tree-Search in Chapel versus CUDA and HIP. *2024 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, San Francisco, USA, 2024, pp. 872-879. DOI: [10.1109/IPDPSW63119.2024.00156](https://doi.org/10.1109/IPDPSW63119.2024.00156).
3. G. Helbecque, E. Krishnasamy, N. Melab, P. Bouvry. GPU Computing in Chapel: Application to Tree-Search Algorithms. *International Conference in Optimization and Learning (OLA 2024)*, Dubrovnik, Croatia, 2024.