cmake_minimum_required(VERSION 3.18)
project(PFSP LANGUAGES C CXX)

# ==== Build options ====
option(USE_HIP "Build with HIP instead of CUDA" OFF)
option(CUDA_ARCH "CUDA architecture (sm_XX)" "80")
option(HIP_ARCH "HIP architecture (gfxXXXX)" "gfx1102")

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ==== Common includes ====
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/lib ${CMAKE_SOURCE_DIR}/../common)

# ==== Dependencies ====
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# ==== Common C sources ====
set(COMMON_LIB_SOURCES
    lib/c_taillard.c
    lib/c_bound_simple.c
    lib/c_bound_johnson.c
    lib/PFSP_node.c
    lib/PFSP_lib.c
    lib/Pool_atom.c
    lib/PFSP_statistic.c
    ../common/util.c
)

# ==== CUDA sources ====
set(CUDA_KERNELS
    ../common/gpu_util.cu
    lib/PFSP_gpu_lib.cu
    lib/bounds_gpu.cu
)

# ==== Sequential executable (C only) ====
add_executable(pfsp_c.out
    pfsp_c.c
    ${COMMON_LIB_SOURCES}
)
target_compile_options(pfsp_c.out PRIVATE -O3 -Wall -g)
target_link_libraries(pfsp_c.out PRIVATE m)

# ==== Multi-core executable (C only) ====
add_executable(pfsp_omp_c.out
    pfsp_omp_c.c
    ${COMMON_LIB_SOURCES}
)
target_compile_options(pfsp_omp_c.out PRIVATE -O3 -Wall -fopenmp -g)
target_link_libraries(pfsp_omp_c.out PRIVATE OpenMP::OpenMP_C m)

# ==== GPU Backend Switch ====

# ==== HIP Backend ====
if(USE_HIP)
    message(STATUS "Building with HIP backend")
    enable_language(HIP)
    find_package(HIP REQUIRED)

    set(CMAKE_HIP_STANDARD 14)
    set(CMAKE_HIP_STANDARD_REQUIRED ON)

    # Hipify CUDA sources into HIP sources in build dir
    set(HIP_SOURCES)
    foreach(src ${CUDA_KERNELS})
        get_filename_component(fname ${src} NAME_WE) # basename without extension
        set(hip_src "${CMAKE_BINARY_DIR}/${fname}.hip")
        add_custom_command(
            OUTPUT ${hip_src}
            COMMAND ${HIP_HIPIFY_PERL} ${CMAKE_SOURCE_DIR}/${src} > ${hip_src}
            DEPENDS ${src}
            COMMENT "Hipifying ${src}"
            VERBATIM
        )
        list(APPEND HIP_SOURCES ${hip_src})
    endforeach()

    # # Single-GPU HIP
    # add_executable(pfsp_gpu_hip.out
    #     pfsp_gpu_cuda.c
    #     ${COMMON_LIB_SOURCES}
    #     ${HIP_SOURCES}
    # )
    # set_target_properties(pfsp_gpu_hip.out PROPERTIES HIP_ARCHITECTURES ${HIP_ARCH})
    # target_compile_options(pfsp_gpu_hip.out PRIVATE -O3 -Wall -g)
    # target_link_libraries(pfsp_gpu_hip.out PRIVATE m)

    # Multi-GPU HIP + OpenMP
    add_executable(pfsp_multigpu_hip.out
        pfsp_multigpu_cuda.c
        ${COMMON_LIB_SOURCES}
        ${HIP_SOURCES}
    )
    set_target_properties(pfsp_multigpu_hip.out PROPERTIES HIP_ARCHITECTURES ${HIP_ARCH})
    target_compile_options(pfsp_multigpu_hip.out PRIVATE -O3 -Wall -fopenmp -g)
    target_link_libraries(pfsp_multigpu_hip.out PRIVATE OpenMP::OpenMP_C m)

    # Distributed Multi-GPU HIP + MPI + OpenMP
    add_executable(pfsp_dist_multigpu_hip.out
        pfsp_dist_multigpu_cuda.c
        ${COMMON_LIB_SOURCES}
        ${HIP_SOURCES}
    )
    set_target_properties(pfsp_dist_multigpu_hip.out PROPERTIES HIP_ARCHITECTURES ${HIP_ARCH})
    target_compile_options(pfsp_dist_multigpu_hip.out PRIVATE -O3 -Wall -fopenmp -g)
    target_link_libraries(pfsp_dist_multigpu_hip.out PRIVATE MPI::MPI_C OpenMP::OpenMP_C m)

# ==== CUDA Backend ====
else()
    message(STATUS "Building with CUDA backend")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # # Single-GPU CUDA
    # add_executable(pfsp_gpu_cuda.out
    #     pfsp_gpu_cuda.c
    #     ${COMMON_LIB_SOURCES}
    #     ${CUDA_KERNELS}
    # )
    # set_target_properties(pfsp_gpu_cuda.out PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    # target_compile_options(pfsp_gpu_cuda.out PRIVATE -O3 -Wall -g)
    # target_link_libraries(pfsp_gpu_cuda.out PRIVATE CUDA::cudart m)

    # Multi-GPU CUDA + OpenMP
    add_executable(pfsp_multigpu_cuda.out
        pfsp_multigpu_cuda.c
        ${COMMON_LIB_SOURCES}
        ${CUDA_KERNELS}
    )
    set_target_properties(pfsp_multigpu_cuda.out PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_compile_options(pfsp_multigpu_cuda.out PRIVATE -O3 -Wall -fopenmp -g)
    target_link_libraries(pfsp_multigpu_cuda.out PRIVATE CUDA::cudart OpenMP::OpenMP_C m)

    # Distributed Multi-GPU CUDA + MPI + OpenMP
    add_executable(pfsp_dist_multigpu_cuda.out
        pfsp_dist_multigpu_cuda.c
        ${COMMON_LIB_SOURCES}
        ${CUDA_KERNELS}
    )
    set_target_properties(pfsp_dist_multigpu_cuda.out PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})
    target_compile_options(pfsp_dist_multigpu_cuda.out PRIVATE -O3 -Wall -fopenmp -g)
    target_link_libraries(pfsp_dist_multigpu_cuda.out PRIVATE CUDA::cudart MPI::MPI_C OpenMP::OpenMP_C m)
endif()