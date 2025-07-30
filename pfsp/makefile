SHELL := /bin/bash

SYSTEM ?= 

# Compilers and common options
C_COMPILER    := gcc
MPI_COMPILER  := mpicc
CUDA_COMPILER := nvcc
HIP_COMPILER  := hipcc
AMD_GPU_ARCH  := gfx1102

# Platform-specific flags and libraries
ifeq ($(SYSTEM), g5k)
  # HIP compiler patch
  HIP_COMPILER  := DEVICE_LIB_PATH=/opt/rocm-4.5.0/amdgcn/bitcode/ hipcc
  # library paths
  MPI_INCLUDE  := -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include
  MPI_LIB      := -L/usr/lib/x86_64-linux-gnu/openmpi/lib
  MPI_FLAGS    := -lmpi_cxx -lmpi -lopen-pal
  CUDA_INCLUDE := -I/share/compilers/nvidia/cuda/12.0/include -I/usr/local/cuda-11.2/targets/x86_64-linux/include
  CUDA_LIB     := -L/usr/local/cuda-11.2/targets/x86_64-linux/lib
  AMD_GPU_ARCH  := gfx906
else ifeq ($(SYSTEM), lumi)
  MPI_INCLUDE  := -I/opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/include
  MPI_LIB      := -L/opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/lib
  MPI_FLAGS    := -lmpi
  AMD_GPU_ARCH := gfx90a
endif

C_COMMON_OPTS    := -O3 -Wall -g -Wno-uninitialized
CUDA_COMMON_OPTS := -O3 -arch=sm_80 # TODO: adapt SM automatically, if possible
HIP_COMMON_OPTS  := $(C_COMMON_OPTS) --offload-arch=$(AMD_GPU_ARCH) -Wno-unused-result

LIBPATH := lib
LIBCOMMON := ../common

# Source files
# C_SOURCES        := pfsp_c.c pfsp_gpu_cuda.c pfsp_multigpu_cuda.c pfsp_dist_multigpu_cuda.c
# C_LIB_SOURCES    := $(LIBPATH)/c_taillard.c $(LIBPATH)/c_bound_simple.c $(LIBPATH)/c_bound_johnson.c $(LIBPATH)/PFSP_lib.c $(LIBPATH)/Pool.c $(LIBPATH)/Pool_ext.c $(LIBPATH)/util.c
# CUDA_LIB_SOURCES := $(LIBPATH)/PFSP_gpu_lib.cu $(LIBPATH)/c_bounds_gpu.cu

# Object files
# C_LIB_OBJECTS    := $(C_LIB_SOURCES:.c=.o)
# CUDA_LIB_OBJECTS := $(CUDA_LIB_SOURCES:.cu=.o)
# HIP_OBJECTS      := $(HIP_SOURCES:hip.cu=hip.o)

# Executable names
EXECUTABLES := pfsp_c.out pfsp_gpu_cuda.out pfsp_multigpu_cuda.out pfsp_dist_multigpu_cuda.out pfsp_gpu_hip.out pfsp_multigpu_hip.out pfsp_dist_multigpu_hip.out

# Build codes
all: $(EXECUTABLES)

# Pattern rule for C library source files
$(LIBPATH)/%.o: $(LIBPATH)/%.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

$(LIBCOMMON)/%.o: $(LIBCOMMON)/%.c
	$(C_COMPILER) $(C_COMMON_OPTS) -c $< -o $@

# Pattern rule for CUDA library source files
$(LIBPATH)/%.o: $(LIBPATH)/%.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@

$(LIBCOMMON)/%.o: $(LIBCOMMON)/%.cu
	$(CUDA_COMPILER) $(CUDA_COMMON_OPTS) -c $< -o $@


# Build executable for sequential in C
pfsp_c.out: pfsp_c.c $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/util.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@ -lm

# Build executable for single-GPU in C+CUDA
pfsp_gpu_cuda.out: pfsp_gpu_cuda.c $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/gpu_util.o $(LIBCOMMON)/util.o $(LIBPATH)/PFSP_gpu_lib.o $(LIBPATH)/PFSP_statistic.o
	$(C_COMPILER) $(C_COMMON_OPTS) $^ -o $@ -lm -lcudart $(CUDA_INCLUDE) $(CUDA_LIB)

# Build executable for multi-GPU in C+OpenMP+CUDA
pfsp_multigpu_cuda.out: pfsp_multigpu_cuda.c $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/util.o $(LIBPATH)/PFSP_gpu_lib.o $(LIBPATH)/PFSP_statistic.o
	$(C_COMPILER) $(C_COMMON_OPTS) -fopenmp $^ -o $@ -lm -lcudart $(CUDA_INCLUDE) $(CUDA_LIB)

# Build executable for distributed multi-GPU in C+MPI+OpenMP+CUDA
pfsp_dist_multigpu_cuda.out: pfsp_dist_multigpu_cuda.c $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/util.o $(LIBPATH)/PFSP_gpu_lib.o $(LIBPATH)/PFSP_statistic.o
	$(MPI_COMPILER) $(C_COMMON_OPTS) -fopenmp $^ -o $@ -lm -lcudart $(CUDA_INCLUDE) $(CUDA_LIB)

$(LIBPATH)/PFSP_gpu_lib_hip.o: $(LIBPATH)/PFSP_gpu_lib.cu
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -c $<.hip -o $@

$(LIBCOMMON)/gpu_util_hip.o: $(LIBCOMMON)/gpu_util.cu
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -c $<.hip -o $@

# TODO: find an elegant way to avoid intermediate *_hip.o object files

# Build executable for single-GPU in C+HIP
pfsp_gpu_hip.o: pfsp_gpu_cuda.c
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -c $<.hip -o $@

pfsp_gpu_hip.out: pfsp_gpu_hip.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/util.o $(LIBCOMMON)/gpu_util_hip.o $(LIBPATH)/PFSP_gpu_lib_hip.o $(LIBPATH)/PFSP_statistic.o
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $^ -o $@

# Build executable for multi-GPU in C+OpenMP+HIP
pfsp_multigpu_hip.o: pfsp_multigpu_cuda.c
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp -c $<.hip -o $@

pfsp_multigpu_hip.out: pfsp_multigpu_hip.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/util.o $(LIBPATH)/PFSP_gpu_lib_hip.o $(LIBPATH)/PFSP_statistic.o
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) -fopenmp $^ -o $@

# Build executable for distributed multi-GPU in C+MPI+OpenMP+HIP
pfsp_dist_multigpu_hip.o: pfsp_dist_multigpu_cuda.c
	hipify-perl $< > $<.hip
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $(MPI_INCLUDE) $(MPI_LIB) -fopenmp -c $<.hip -o $@

pfsp_dist_multigpu_hip.out: pfsp_dist_multigpu_hip.o $(LIBPATH)/c_taillard.o $(LIBPATH)/c_bound_simple.o $(LIBPATH)/c_bound_johnson.o $(LIBPATH)/PFSP_node.o $(LIBPATH)/PFSP_lib.o $(LIBPATH)/Pool_atom.o $(LIBCOMMON)/util.o $(LIBPATH)/PFSP_gpu_lib_hip.o $(LIBPATH)/PFSP_statistic.o
	$(HIP_COMPILER) $(HIP_COMMON_OPTS) $(MPI_INCLUDE) $(MPI_LIB) $(MPI_FLAGS) -fopenmp $^ -o $@

# Utilities
.PHONY: clean

clean:
	rm -f *.out *.o *.hip $(LIBPATH)/*.o $(LIBCOMMON)/*.o $(LIBPATH)/*.hip $(LIBCOMMON)/*.hip