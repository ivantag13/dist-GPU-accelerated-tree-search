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
# message(STATUS "MPI_C_FOUND: ${MPI_C_FOUND}")
# message(STATUS "MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}") 
# message(STATUS "MPI_C_COMPILE_OPTIONS: ${MPI_C_COMPILE_OPTIONS}")
# message(STATUS "MPI_C_LINK_FLAGS: ${MPI_C_LINK_FLAGS}")
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

# ==== CUDA kernel sources ====
set(CUDA_KERNELS
    ../common/gpu_util.cu
    lib/PFSP_gpu_lib.cu
    lib/bounds_gpu.cu
)

# ==== CUDA main sources ====
set(CUDA_MAIN_SOURCES
    pfsp_multigpu_cuda.c
    pfsp_dist_multigpu_cuda.c
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
    set(HIP_KERNEL_SOURCES)
    foreach(src ${CUDA_KERNELS})
        get_filename_component(fname ${src} NAME_WE)
        set(hip_src "${CMAKE_BINARY_DIR}/${fname}.hip")
        add_custom_command(
            OUTPUT ${hip_src}
            COMMAND hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/${src} > ${hip_src}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            COMMENT "Hipifying ${src}"
            VERBATIM
        )
        list(APPEND HIP_KERNEL_SOURCES ${hip_src})
    endforeach()

    # Hipify main sources separately
    set(HIP_MULTIGPU_MAIN "${CMAKE_BINARY_DIR}/pfsp_multigpu_cuda.hip")
    add_custom_command(
        OUTPUT ${HIP_MULTIGPU_MAIN}
        COMMAND hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/pfsp_multigpu_cuda.c > ${HIP_MULTIGPU_MAIN}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pfsp_multigpu_cuda.c
        COMMENT "Hipifying pfsp_multigpu_cuda.c"
        VERBATIM
    )

    set(HIP_DIST_MULTIGPU_MAIN "${CMAKE_BINARY_DIR}/pfsp_dist_multigpu_cuda.hip")
    add_custom_command(
        OUTPUT ${HIP_DIST_MULTIGPU_MAIN}
        COMMAND hipify-perl ${CMAKE_CURRENT_SOURCE_DIR}/pfsp_dist_multigpu_cuda.c > ${HIP_DIST_MULTIGPU_MAIN}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/pfsp_dist_multigpu_cuda.c
        COMMENT "Hipifying pfsp_dist_multigpu_cuda.c"
        VERBATIM
    )

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
        ${HIP_MULTIGPU_MAIN}
        ${COMMON_LIB_SOURCES}
        ${HIP_KERNEL_SOURCES}
    )
    set_target_properties(pfsp_multigpu_hip.out PROPERTIES HIP_ARCHITECTURES ${HIP_ARCH})
    target_compile_options(pfsp_multigpu_hip.out PRIVATE -O3 -Wall -fopenmp -g)
    target_link_options(pfsp_multigpu_hip.out PRIVATE -fopenmp)
    target_link_libraries(pfsp_multigpu_hip.out PRIVATE OpenMP::OpenMP_C m)

    # Distributed Multi-GPU HIP + MPI + OpenMP
    add_executable(pfsp_dist_multigpu_hip.out
        ${HIP_DIST_MULTIGPU_MAIN}
        ${COMMON_LIB_SOURCES}
        ${HIP_KERNEL_SOURCES}
    )
    set_target_properties(pfsp_dist_multigpu_hip.out PROPERTIES HIP_ARCHITECTURES ${HIP_ARCH})
    target_compile_options(pfsp_dist_multigpu_hip.out PRIVATE -O3 -Wall -fopenmp -g)
    target_link_options(pfsp_dist_multigpu_hip.out PRIVATE -fopenmp)
    
    # Manually discover MPI include paths
    execute_process(
        COMMAND ${MPI_C_COMPILER} -show
        OUTPUT_VARIABLE MPI_COMPILER_SHOW
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MPI_COMPILER_RESULT
    )

    if(MPI_COMPILER_RESULT EQUAL 0)
        # Parse the output to find include directories
        string(REGEX MATCHALL "-I[^ ]+" MPI_INCLUDE_FLAGS "${MPI_COMPILER_SHOW}")
        if(MPI_INCLUDE_FLAGS)
            # Remove the -I prefix and add as include directories
            foreach(include_flag ${MPI_INCLUDE_FLAGS})
                string(SUBSTRING "${include_flag}" 2 -1 include_dir)
                target_include_directories(pfsp_dist_multigpu_hip.out PRIVATE ${include_dir})
                message(STATUS "Adding MPI include directory: ${include_dir}")
            endforeach()
        else()
            # If no -I flags found, try common Cray MPI paths
            target_include_directories(pfsp_dist_multigpu_hip.out PRIVATE 
                /opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/include
                /opt/cray/pe/mpi/17.0/include
            )
            message(STATUS "Using default Cray MPI include paths")
        endif()
    else()
        # Fallback to common Cray MPI paths
        target_include_directories(pfsp_dist_multigpu_hip.out PRIVATE 
            /opt/cray/pe/mpich/8.1.29/ofi/cray/17.0/include
            /opt/cray/pe/mpi/17.0/include
        )
        message(STATUS "Using fallback Cray MPI include paths")
    endif()
    
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