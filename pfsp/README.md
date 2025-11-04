# Building code with CMake file

One can also build with the CMake file option. The CUDA option is completely functional. The HIP option is under process of fixing. One only needs to run on the command line for the CUDA configuration step (`-B` build directory, `-S` source directory):
`cmake -B build -S . -DUSE_HIP=OFF -DCUDA_ARCH=80`

For the HIP configuration step (still unstable):
`cmake -B build -S . -DUSE_HIP=ON -DHIP_ARCH=gfx90a`

For the build step (`-j` build in parallel):
`cmake --build build -j`

# Building code with makefile option

The `makefile` builds all CUDA programs and their HIP versions (using ROCm/HIP `hipify-perl` tool). From this version onward, the pure single-GPU version, `pfsp_gpu_cuda.c`, will be deprecated serving mainly for development purposes.

Its `SYSTEM` command-line option can be set to `{g5k, lumi}` to manually handle the system specific library paths. It defaults to no system for local compilation. Documentation for supported systems:
- the [Grid5000](https://www.grid5000.fr/w/Grid5000:Home) large-scale testbed;
- the [LUMI](https://docs.lumi-supercomputer.eu/) pre-exascale supercomputer.

Suppose you want to compile the distributed version of the code on the Grid5000 (chuc node). The following command should be executed:

`make SYSTEM=g5k pfsp_dist_multigpu_cuda.out`

# Running your code

This folder contains a subfolder called `launch_scripts` (The file `sgpu_launch.sh` is deprecated). It contains bash scripts to launch batch of experiments in LUMI supercomputer (SLURM system):
- `mgpu_launch.sh`: launches sets of experiments for a multi-core multi-GPU implementation. It contains the following flags:
    - `-m`: sets the parameter `m`, a threshold to determine start shared-memory multiprocessing directives (default set to `25`)
    - `-M`: sets the parameter `M`, a threshold to determine maximum amount of nodes for GPU offloading (default set to `50000`)
    - `-T`: sets the parameter `T`, a threshold to determine maximum amount of nodes for each CPU parallel amount of maximum nodes treated (default set to `5000`)
    - `-r`: parameter that determines the amount of repetitions par iteration (default set to `1`)
    - `-j`: parameter that determine the number of jobs for launching a certain class of Taillard instances (obligatory for proper functioning of this bash file, options for jobs are `20`, `50`, `100`, `200`, `500`)
    - `-g`: parameter that determines the amount of machines for launching a certain class of Taillard instances (obligatory for proper functioning of this bash file, options for jobs are `5`, `10`, `20`)
    - `-l`: sets bounding function `l` (do not choose value `0` because of malfunctioning on GPU devices - check issue #4).
    - `-w`: enables (or not) intra-node work stealing (default set to `1`)
    - `-D`: sets the amount of GPU devices (default set to `1`)
    - `-C`: sets for additional multi-core processing (default set to `1` - activated multi-core). If multi-core is activated the mapping and number of extra CPU processing units is automatically determined in our implementation.

- `dmgpu_launch.sh `: launches sets of experiments for the distributed multi-GPU implementation. It contains all flags of `mgpu_launch.sh` (with exception of `-w`) plus:
    - `-L`: enables inter-node load balancing (default set to `1` - work stealing. Other option is `0` - static/no load balancing)
    - `-n`: sets the amount of MPI processes (default set to `1` - one MPI process per compute node)

When running `dmgpu_launch.sh`, one should launch it by setting the amount of nodes for the `sbatch` command, `--nodes=`, equal as the parameter `-n`. The use of the parameter `C` in the distributed version is unstable, please keep to the default setting. For instance, if one want to launch experiments for instances with `20` jobs and `10` machines, with `6` GPUs, lower bound function `lb2`, inter-node load-balancing being work stealing, and with `16` MPI processes one should run:

`sbatch --nodes=16 dmgpu_launch.sh -j 20 -g 10 -D 6 -l 2 -L 1 -n 16`

**Note**: For executing instances with `50` jobs or more, one should edit (for now) the file `lib/macro.h` for setting macro `MAX_JOBS` as the amount of jobs for desired instances (by default is set to `20`). Furthermore, one could also optimize by changing the variable `MAX_MACHINES` (set to `20` by default) to correspond the number of machines chosen for a given execution of a certain Taillard instance before compilation.