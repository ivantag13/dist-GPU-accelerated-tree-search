# Building code

The provided `makefile` builds all CUDA-based codes as well as HIP-based ones (using the ROCm/HIP `hipify-perl` tool).

The `SYSTEM` command-line option can be set to `{g5k, lumi}` to handle manually the system specific library paths etc. It defaults to `g5k` compilation. For the moment, the following systems are supported:
- the [Grid5000](https://www.grid5000.fr/w/Grid5000:Home) large-scale testbed;
- the [LUMI](https://docs.lumi-supercomputer.eu/) pre-exascale supercomputer.

This folder contains a subfolder called `launch_scripts` which contains bash scripts to launch batch of experiments to LUMI supercomputer (SLURM system):
- `sgpu_launch.sh`: launches sets of experiments for the single-GPU implementation. It contains the following flags:
    - `-m`: for setting the parameter `m`, threshold to determine start of GPU offloading (default set to `25`)
    - `-M`: for setting the parameter `M`, threshold to determine maximum amount of nodes for GPU offloading (default set to `50000`)
    - `-r`: parameter that determines the amount of repetitions par iteration (default set to `1`)
    - `-j`: represents the parameter jobs for launching a certain class of Taillard instances (obligatory for proper functioning of this bash file, options for jobs are `20`, `50`, `100`, `200`, `500` - check issue #2)
    - `-g`: represents the parameter machines for launching a certain class of Taillard instances (obligatory for proper functioning of this bash file, options for jobs are `5`, `10`, `20` - check issue #2)
    - `-l`: sets bounding function `l`.

- `mgpu_launch.sh`: launches sets of experiments for the multi-GPU implementation. It contains all flags of `sgpu_launch.sh` plus:
    - `-w`: enables (or not) intra-node work stealing (default set to `1`)
    - `-D`: sets the amount of GPU devices (default set to `1`)

- `dmgpu_launch.sh `: launches sets of experiments for the distributed multi-GPU implementation. It contains all flags of `mgpu_launch.sh` (with exception of `-w`) and `sgpu_launch.sh` plus:
    - `-L`: enables inter-node load balancing (default set to `0`, which is static, other options are `1`, work sharing, and `2`, work stealing)
    - `-n`: sets the amount of MPI processes (default set to `1`)

When running `dmgpu_launch.sh`, one should launch by setting the amount of nodes for the `sbatch` command equal as the parameter `-n`. For instance, if one want to launch experiments for instances with `20` jobs and `10` machines, with `6` GPUs, lower bound function `lb2`, inter-node load-balancing being work sharing, and with `16` MPI processes one should run:

`sbatch --nodes=16 dmgpu_launch.sh -j 20 -g 10 -D 6 -l 2 -L 1 -n 16`

**Note**: For executing instances with `50` jobs or more, one should edit (for now) the file `lib/macro.h` for setting macro `MAX_JOBS` as the amount of jobs for desired instances (by default is set to `20`)