ginkgo_create_mpi_test(array 2)
ginkgo_create_mpi_test(exception_helpers 2)
ginkgo_create_mpi_test(mpi_executor 4)
ginkgo_create_mpi_test(multi_rank_tests 4)

if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda)
endif()
