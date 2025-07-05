#!/bin/bash
module load gcc/12.2.0_gcc-10.4.0 cuda/12.0.0_gcc-10.4.0 hip/5.2.0_gcc-10.4.0
module load rocm-openmp-extras/5.2.0_gcc-10.4.0
sudo-g5k apt-get install libomp-dev -y
find /usr/lib/ /opt/rocm/ -name libomp.so
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH' >> ~/.bashrc 
source ~/.bashrc