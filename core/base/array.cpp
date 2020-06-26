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
    using itype = int;
    auto mpi_exec = as<gko::MpiExecutor>(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();

    // int because MPI functions only support 32 bit integers.
    itype num_subsets = index_set.get_num_subsets();
    auto num_subsets_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather<itype, itype>(&num_subsets, 1,
                                   num_subsets_array.get_data(), 1, root_rank);
    auto total_num_subsets =
        std::accumulate(num_subsets_array.get_data(),
                        num_subsets_array.get_data() + num_ranks, 0);
    itype num_elems = static_cast<itype>(index_set.get_num_elems());
    auto num_elems_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather<itype, itype>(&num_elems, 1, num_elems_array.get_data(), 1,
                                   root_rank);

    auto start_idx_array = Array<itype>{sub_exec->get_master(),
                                        static_cast<size_type>(num_subsets)};
    auto num_elems_in_subset = Array<itype>{
        sub_exec->get_master(), static_cast<size_type>(num_subsets)};
    auto offset_array =
        Array<itype>{sub_exec->get_master(),
                     static_cast<size_type>(num_ranks * total_num_subsets)};
    auto global_num_elems_subset_array =
        Array<itype>{sub_exec->get_master(),
                     static_cast<size_type>(num_ranks * total_num_subsets)};
    auto first_interval = (index_set.get_first_interval());
    for (auto i = 0; i < num_subsets; ++i) {
        start_idx_array.get_data()[i] = *(*first_interval).begin();
        num_elems_in_subset.get_data()[i] = (*first_interval).get_num_elems();
        first_interval++;
    }
    auto displ = gko::Array<itype>{num_subsets_array};
    std::partial_sum(displ.get_data(), displ.get_data() + displ.get_num_elems(),
                     displ.get_data());
    for (auto i = 0; i < displ.get_num_elems(); ++i) {
        displ.get_data()[i] -= num_subsets_array.get_data()[i];
    }
    mpi_exec->gather<itype, itype>(
        start_idx_array.get_data(), num_subsets, offset_array.get_data(),
        num_subsets_array.get_data(), displ.get_data(), root_rank);
    mpi_exec->gather<itype, itype>(num_elems_in_subset.get_data(), num_subsets,
                                   global_num_elems_subset_array.get_data(),
                                   num_subsets_array.get_data(),
                                   displ.get_data(), root_rank);

    auto tag = gko::Array<itype>{sub_exec->get_master(),
                                 static_cast<size_type>(num_subsets)};
    for (auto t = 0; t < num_subsets; ++t) {
        tag.get_data()[t] = (my_rank + 1) * 1e4 + t;
    }
    auto tags = gko::Array<itype>{
        sub_exec->get_master(),
        static_cast<size_type>(num_ranks * total_num_subsets)};
    mpi_exec->gather<itype, itype>(tag.get_data(), num_subsets, tags.get_data(),
                                   num_subsets_array.get_data(),
                                   displ.get_data(), root_rank);

    auto distributed_array = Array{exec, index_set.get_num_elems()};
    auto idx = 0;
    for (auto in_rank = 0; in_rank < num_ranks; ++in_rank) {
        auto n_subsets = num_subsets_array.get_data()[in_rank];
        if (in_rank != root_rank) {
            if (my_rank == root_rank) {
                for (auto in_subset = 0; in_subset < n_subsets; ++in_subset) {
                    auto offset = offset_array.get_data()[idx];
                    auto g_n_elems =
                        global_num_elems_subset_array.get_data()[idx];
                    mpi_exec->send(&data_.get()[offset], g_n_elems, in_rank,
                                   tags.get_data()[idx]);
                    idx++;
                }
            }
        } else {
            idx += n_subsets;
        }
    }
    auto offset = 0;
    for (auto in_subset = 0; in_subset < num_subsets; ++in_subset) {
        auto n_elems = num_elems_in_subset.get_data()[in_subset];
        auto start_idx = start_idx_array.get_data()[in_subset];
        if (my_rank != root_rank) {
            mpi_exec->recv(&distributed_array.get_data()[offset], n_elems,
                           root_rank, tag.get_data()[in_subset]);
        } else {
            distributed_array.get_executor()->get_mem_space()->copy_from(
                sub_exec->get_mem_space().get(), n_elems,
                &data_.get()[start_idx], &distributed_array.get_data()[offset]);
        }
        offset += n_elems;
    }

    return std::move(distributed_array);
}


#define GKO_DECLARE_ARRAY_DISTRIBUTE(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::distribute_data(    \
        std::shared_ptr<gko::Executor> exec,               \
        const IndexSet<IndexType> &index_set)

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ARRAY_DISTRIBUTE);

}  // namespace gko
