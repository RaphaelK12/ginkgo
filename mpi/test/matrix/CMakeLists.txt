ginkgo_create_mpi_test(distributed_dense 2)
ginkgo_create_mpi_test(distributed_coo 2)
ginkgo_create_mpi_test(distributed_csr 2)

if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda)
endif()
