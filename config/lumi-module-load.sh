#!/bin/bash
module load LUMI/24.03 partition/G buildtools/24.03
module load rocm/6.0.3
module load cray-mpich/8.1.29
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm-6.0.3/lib/llvm/lib
