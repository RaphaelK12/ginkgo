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

#ifndef GKO_MPI_BINDINGS_HPP_
#define GKO_MPI_BINDINGS_HPP_


#include <mpi.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include <iostream>

namespace gko {
/**
 * @brief The bindings namespace.
 *
 * @ingroup bindings
 */
namespace bindings {
/**
 * @brief The MPI namespace.
 *
 * @ingroup mpi
 */
namespace mpi {


inline MPI_Comm create_comm(MPI_Comm &comm_in, int color, int key)
{
    MPI_Comm comm_out;
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_split(comm_in, color, key, &comm_out));
    return comm_out;
}


inline MPI_Comm *get_comm_world()
{
    MPI_Comm comm = MPI_COMM_WORLD;
    return std::move(&comm);
}


inline void free_comm(MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_free(&comm));
}


inline void wait(MPI_Request *request, MPI_Status *status)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Wait(request, status));
}


inline void send(const void *send_buffer, const int send_count,
                 MPI_Datatype &send_type, const int destination_rank,
                 const int send_tag, MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Send(send_buffer, send_count, send_type,
                                      destination_rank, send_tag, comm));
}


inline void recv(void *recv_buffer, const int recv_count,
                 MPI_Datatype &recv_type, const int source_rank,
                 const int recv_tag, MPI_Comm &comm, MPI_Status *status)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Recv(recv_buffer, recv_count, recv_type,
                                      source_rank, recv_tag, comm, status));
}


inline void i_send(const void *send_buffer, const int send_count,
                   MPI_Datatype &send_type, const int destination_rank,
                   const int send_tag, MPI_Comm &comm, MPI_Request *request)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Isend(send_buffer, send_count, send_type,
                                       destination_rank, send_tag, comm,
                                       request));
}


inline void i_recv(void *recv_buffer, const int recv_count,
                   MPI_Datatype &recv_type, const int source_rank,
                   const int recv_tag, MPI_Comm &comm, MPI_Request *request)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Irecv(recv_buffer, recv_count, recv_type,
                                       source_rank, recv_tag, comm, request));
}


inline void broadcast(void *buffer, int count, MPI_Datatype &bcast_type,
                      int root_rank, MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Bcast(buffer, count, bcast_type, root_rank, comm));
}


inline void reduce(const void *send_buffer, void *recv_buffer, int count,
                   MPI_Datatype &reduce_type, MPI_Op operation, int root_rank,
                   MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Reduce(send_buffer, recv_buffer, count,
                                        reduce_type, operation, root_rank,
                                        comm));
}


inline void all_reduce(const void *send_buffer, void *recv_buffer, int count,
                       MPI_Datatype &reduce_type, MPI_Op operation,
                       MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Allreduce(send_buffer, recv_buffer, count,
                                           reduce_type, operation, comm));
}


inline void i_reduce(const void *send_buffer, void *recv_buffer, int count,
                     MPI_Datatype &reduce_type, MPI_Op operation, int root_rank,
                     MPI_Comm &comm, MPI_Request *requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Ireduce(send_buffer, recv_buffer, count,
                                         reduce_type, operation, root_rank,
                                         comm, requests));
}


inline void i_all_reduce(const void *send_buffer, void *recv_buffer, int count,
                         MPI_Datatype &reduce_type, MPI_Op operation,
                         MPI_Comm &comm, MPI_Request *requests)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Iallreduce(send_buffer, recv_buffer, count,
                                            reduce_type, operation, comm,
                                            requests));
}


inline void gather(const void *send_buffer, const int send_count,
                   MPI_Datatype &send_type, void *recv_buffer,
                   const int recv_count, MPI_Datatype &recv_type, int root,
                   MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Gather(send_buffer, send_count, send_type,
                                        recv_buffer, recv_count, recv_type,
                                        root, comm));
}

inline void gatherv(const void *send_buffer, const int send_count,
                    MPI_Datatype &send_type, void *recv_buffer,
                    const int *recv_counts, const int *displacements,
                    MPI_Datatype &recv_type, int root_rank, MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Gatherv(send_buffer, send_count, send_type, recv_buffer,
                    recv_counts, displacements, recv_type, root_rank, comm));
}

inline void scatter(const void *send_buffer, const int send_count,
                    MPI_Datatype &send_type, void *recv_buffer,
                    const int recv_count, MPI_Datatype &recv_type, int root,
                    MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(MPI_Scatter(send_buffer, send_count, send_type,
                                         recv_buffer, recv_count, recv_type,
                                         root, comm));
}

inline void scatterv(const void *send_buffer, const int *send_counts,
                     const int *displacements, MPI_Datatype &send_type,
                     void *recv_buffer, const int recv_count,
                     MPI_Datatype &recv_type, int root_rank, MPI_Comm &comm)
{
    GKO_ASSERT_NO_MPI_ERRORS(
        MPI_Scatterv(send_buffer, send_counts, displacements, send_type,
                     recv_buffer, recv_count, recv_type, root_rank, comm));
}


}  // namespace mpi
}  // namespace bindings
}  // namespace gko


#endif  // GKO_MPI_BINDINGS_HPP_
