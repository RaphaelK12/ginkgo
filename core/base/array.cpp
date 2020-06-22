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

#include <ginkgo/core/base/array.hpp>


#include <ginkgo/core/base/math.hpp>


#include "core/components/precision_conversion.hpp"


namespace gko {
namespace conversion {


GKO_REGISTER_OPERATION(convert, components::convert_precision);


}  // namespace conversion


namespace detail {


template <typename SourceType, typename TargetType>
void convert_data(std::shared_ptr<const Executor> exec, size_type size,
                  const SourceType *src, TargetType *dst)
{
    exec->run(conversion::make_convert(size, src, dst));
}


#define GKO_DECLARE_ARRAY_CONVERSION(From, To)                              \
    void convert_data<From, To>(std::shared_ptr<const Executor>, size_type, \
                                const From *, To *)

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_ARRAY_CONVERSION);


}  // namespace detail

template <typename ValueType>
template <typename IndexType>
Array<ValueType> Array<ValueType>::distribute_data(
    std::shared_ptr<gko::Executor> exec, const IndexSet<IndexType> &index_set)
{
    GKO_ASSERT_MPI_EXEC(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto mpi_exec = dynamic_cast<gko::MpiExecutor *>(exec.get());
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();
    auto comm = mpi_exec->get_communicator();

    // int because MPI functions only support 32 bit integers.
    auto num_elems = static_cast<int>(index_set.get_num_elements());
    auto distributed_array = Array{exec, index_set.get_num_elements()};
    auto send_counts =
        Array<int>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather<int, int>(&num_elems, 1, send_counts.get_data(), 1,
                               root_rank);
    auto displacements = Array<int>{send_counts};
    // IMPROVE-ME
    std::partial_sum(displacements.get_data(),
                     displacements.get_data() + displacements.get_num_elems(),
                     displacements.get_data());
    for (auto i = 0; i < displacements.get_num_elems(); ++i) {
        displacements.get_data()[i] -= send_counts.get_data()[i];
    }
    mpi_exec->scatter<ValueType, ValueType>(
        data_.get(), send_counts.get_data(), displacements.get_data(),
        distributed_array.get_data(), num_elems, root_rank);

    return std::move(distributed_array);
}

#define GKO_DECLARE_ARRAY_DISTRIBUTE(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::distribute_data(    \
        std::shared_ptr<gko::Executor> exec,               \
        const IndexSet<IndexType> &index_set)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARRAY_DISTRIBUTE);

}  // namespace gko
