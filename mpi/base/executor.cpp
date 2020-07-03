/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include <iostream>
#include <map>


#include "mpi/base/mpi_bindings.hpp"

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "mpi/base/mpi_helpers.hpp"


namespace gko {


void MpiExecutor::mpi_init()
{
    auto flag = MpiExecutor::is_initialized();
    if (!flag) {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Init_thread(
            &(this->num_args_), &(this->args_), this->required_thread_support_,
            &(this->provided_thread_support_)));
    }
    this->mpi_comm_ = MPI_COMM_WORLD;
}


void MpiExecutor::create_sub_executors(
    std::vector<std::string> &sub_exec_list,
    std::shared_ptr<gko::Executor> &sub_executor)
{
    auto num_gpus = this->get_num_gpus();
    int dev_id = 0;
    for (auto i = 0; i < sub_exec_list.size(); ++i) {
        if (sub_exec_list[i] == "omp") {
            sub_executor = gko::OmpExecutor::create();
        }
        if (sub_exec_list[i] == "reference") {
            sub_executor = gko::ReferenceExecutor::create();
        }
        if (sub_exec_list[i] == "cuda" && num_gpus > 0 && dev_id < num_gpus) {
            sub_executor =
                gko::CudaExecutor::create(dev_id, gko::OmpExecutor::create());
            dev_id++;
        }
        if (sub_exec_list[i] == "hip" && num_gpus > 0 && dev_id < num_gpus) {
            sub_executor =
                gko::HipExecutor::create(dev_id, gko::OmpExecutor::create());
            dev_id++;
        }
    }
}


MpiExecutor::request_manager<MPI_Request> MpiExecutor::create_requests_array(
    int size)
{
    return MpiExecutor::request_manager<MPI_Request>{
        new MPI_Request[size], [size](MPI_Request *req) { delete req; }};
}


void MpiExecutor::synchronize_communicator(MPI_Comm &comm) const
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(comm));
}


void MpiExecutor::synchronize() const
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Barrier(MPI_COMM_WORLD));
}


// MPI_Op MpiExecutor::create_operation(
//     std::function<void(void *, void *, int *, MPI_Datatype *)> func, void
//     *arg1, void *arg2, int *len, MPI_Datatype *type)
// {
//     MPI_Op operation;
//     bindings::mpi::create_op(
//         func.target<void(void *, void *, int *, MPI_Datatype *)>(), true,
//         &operation);
//     return operation;
// }


int MpiExecutor::get_my_rank() const
{
    auto my_rank = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(MPI_COMM_WORLD, &my_rank));
    return my_rank;
}


int MpiExecutor::get_num_ranks() const
{
    int size = 1;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_size(MPI_COMM_WORLD, &size));
    return size;
}


template <typename SendType>
void MpiExecutor::send(const SendType *send_buffer, const int send_count,
                       const int destination_rank, const int send_tag,
                       bool non_blocking)
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    if (!non_blocking) {
        bindings::mpi::send(send_buffer, send_count, send_type,
                            destination_rank, send_tag, this->mpi_comm_);
    } else {
        bindings::mpi::i_send(send_buffer, send_count, send_type,
                              destination_rank, send_tag, this->mpi_comm_,
                              nullptr);
    }
}


template <typename RecvType>
void MpiExecutor::recv(RecvType *recv_buffer, const int recv_count,
                       const int source_rank, const int recv_tag,
                       bool non_blocking)
{
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    if (!non_blocking) {
        bindings::mpi::recv(recv_buffer, recv_count, recv_type, source_rank,
                            recv_tag, this->mpi_comm_, this->mpi_status_.get());
    } else {
        bindings::mpi::i_recv(recv_buffer, recv_count, recv_type, source_rank,
                              recv_tag, this->mpi_comm_, nullptr);
    }
}


template <typename BroadcastType>
void MpiExecutor::broadcast(BroadcastType *buffer, int count, int root_rank)
{
    auto bcast_type = helpers::mpi::get_mpi_type(buffer[0]);
    bindings::mpi::broadcast(buffer, count, bcast_type, root_rank,
                             this->mpi_comm_);
}


template <typename ReduceType>
void MpiExecutor::reduce(const ReduceType *send_buffer, ReduceType *recv_buffer,
                         int count, mpi::op_type op_enum, int root_rank,
                         bool non_blocking)
{
    auto operation = helpers::mpi::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    if (!non_blocking) {
        bindings::mpi::reduce(send_buffer, recv_buffer, count, reduce_type,
                              operation, root_rank, this->mpi_comm_);
    } else {
        bindings::mpi::i_reduce(send_buffer, recv_buffer, count, reduce_type,
                                operation, root_rank, this->mpi_comm_, nullptr);
    }
}


template <typename ReduceType>
void MpiExecutor::all_reduce(const ReduceType *send_buffer,
                             ReduceType *recv_buffer, int count,
                             mpi::op_type op_enum, bool non_blocking)
{
    auto operation = helpers::mpi::get_operation<ReduceType>(op_enum);
    auto reduce_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    if (!non_blocking) {
        bindings::mpi::all_reduce(send_buffer, recv_buffer, count, reduce_type,
                                  operation, this->mpi_comm_);
    } else {
        bindings::mpi::i_all_reduce(send_buffer, recv_buffer, count,
                                    reduce_type, operation, this->mpi_comm_,
                                    nullptr);
    }
}


template <typename SendType, typename RecvType>
void MpiExecutor::gather(const SendType *send_buffer, const int send_count,
                         RecvType *recv_buffer, const int recv_count,
                         int root_rank)
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::gather(send_buffer, send_count, send_type, recv_buffer,
                          recv_count, recv_type, root_rank, this->mpi_comm_);
}


template <typename SendType, typename RecvType>
void MpiExecutor::gather(const SendType *send_buffer, const int send_count,
                         RecvType *recv_buffer, const int *recv_counts,
                         const int *displacements, int root_rank)
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::gatherv(send_buffer, send_count, send_type, recv_buffer,
                           recv_counts, displacements, recv_type, root_rank,
                           this->mpi_comm_);
}


template <typename SendType, typename RecvType>
void MpiExecutor::scatter(const SendType *send_buffer, const int send_count,
                          RecvType *recv_buffer, const int recv_count,
                          int root_rank)
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::scatter(send_buffer, send_count, send_type, recv_buffer,
                           recv_count, recv_type, root_rank, this->mpi_comm_);
}


template <typename SendType, typename RecvType>
void MpiExecutor::scatter(const SendType *send_buffer, const int *send_counts,
                          const int *displacements, RecvType *recv_buffer,
                          const int recv_count, int root_rank)
{
    auto send_type = helpers::mpi::get_mpi_type(send_buffer[0]);
    auto recv_type = helpers::mpi::get_mpi_type(recv_buffer[0]);
    bindings::mpi::scatterv(send_buffer, send_counts, displacements, send_type,
                            recv_buffer, recv_count, recv_type, root_rank,
                            this->mpi_comm_);
}


std::shared_ptr<MpiExecutor> MpiExecutor::create(
    std::initializer_list<std::string> sub_exec_list, int num_args, char **args)
{
    return std::shared_ptr<MpiExecutor>(
        new MpiExecutor(sub_exec_list, num_args, args),
        [](MpiExecutor *exec) { delete exec; });
}


std::shared_ptr<MpiExecutor> MpiExecutor::create()
{
    int num_args = 0;
    char **args;
    return MpiExecutor::create({"reference"}, num_args, args);
}


bool MpiExecutor::is_initialized() const
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Initialized(&flag));
    return flag;
}


bool MpiExecutor::is_finalized() const
{
    int flag = 0;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalized(&flag));
    return flag;
}


void MpiExecutor::destroy()
{
    auto flag = MpiExecutor::is_finalized();
    if (!flag) {
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Finalize());
    }
}


MPI_Comm MpiExecutor::create_communicator(MPI_Comm &comm_in, int color, int key)
{
    return bindings::mpi::create_comm(comm_in, color, key);
}


#define GKO_DECLARE_SEND(SendType)                                            \
    void MpiExecutor::send(const SendType *send_buffer, const int send_count, \
                           const int destination_rank, const int send_tag,    \
                           bool non_blocking)

GKO_INSTANTIATE_FOR_EACH_SEPARATE_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SEND);


#define GKO_DECLARE_RECV(RecvType)                                      \
    void MpiExecutor::recv(RecvType *recv_buffer, const int recv_count, \
                           const int source_rank, const int recv_tag,   \
                           bool non_blocking)

GKO_INSTANTIATE_FOR_EACH_SEPARATE_VALUE_AND_INDEX_TYPE(GKO_DECLARE_RECV);


#define GKO_DECLARE_BCAST(BroadcastType) \
    void MpiExecutor::broadcast(BroadcastType *buffer, int count, int root_rank)

GKO_INSTANTIATE_FOR_EACH_SEPARATE_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCAST);


#define GKO_DECLARE_REDUCE(ReduceType)                                     \
    void MpiExecutor::reduce(                                              \
        const ReduceType *send_buffer, ReduceType *recv_buffer, int count, \
        mpi::op_type operation, int root_rank, bool non_blocking)

GKO_INSTANTIATE_FOR_EACH_SEPARATE_VALUE_AND_INDEX_TYPE(GKO_DECLARE_REDUCE);


#define GKO_DECLARE_ALLREDUCE(ReduceType)                            \
    void MpiExecutor::all_reduce(const ReduceType *send_buffer,      \
                                 ReduceType *recv_buffer, int count, \
                                 mpi::op_type operation, bool non_blocking)

GKO_INSTANTIATE_FOR_EACH_SEPARATE_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ALLREDUCE);


#define GKO_DECLARE_GATHER1(SendType, RecvType)                           \
    void MpiExecutor::gather(const SendType *send_buffer,                 \
                             const int send_count, RecvType *recv_buffer, \
                             const int recv_count, int root_rank)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER1);


#define GKO_DECLARE_GATHER2(SendType, RecvType)                                \
    void MpiExecutor::gather(const SendType *send_buffer,                      \
                             const int send_count, RecvType *recv_buffer,      \
                             const int *recv_counts, const int *displacements, \
                             int root_rank)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_GATHER2);


#define GKO_DECLARE_SCATTER1(SendType, RecvType)                           \
    void MpiExecutor::scatter(const SendType *send_buffer,                 \
                              const int send_count, RecvType *recv_buffer, \
                              const int recv_count, int root_rank)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER1);


#define GKO_DECLARE_SCATTER2(SendType, RecvType)                               \
    void MpiExecutor::scatter(const SendType *send_buffer,                     \
                              const int *send_counts,                          \
                              const int *displacements, RecvType *recv_buffer, \
                              const int recv_count, int root_rank)

GKO_INSTANTIATE_FOR_EACH_COMBINED_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SCATTER2);


}  // namespace gko
