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
#include "core/components/sqrt_array.hpp"


namespace gko {
namespace conversion {


GKO_REGISTER_OPERATION(convert, components::convert_precision);


}  // namespace conversion


namespace utils {


GKO_REGISTER_OPERATION(sqrt, components::sqrt_array);


}  // namespace utils


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


template <typename SourceType>
template <typename TargetType>
void Array<SourceType>::sqrt(Array<TargetType> &sqrt_array)
{
    auto size = this->get_num_elems();
    this->get_executor()->run(
        utils::make_sqrt(size, this->get_const_data(), sqrt_array.get_data()));
}


#define GKO_DECLARE_ARRAY_SQRT(SourceType, TargetType) \
    void Array<SourceType>::sqrt(Array<TargetType> &sqrt_array)

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_PAIR(GKO_DECLARE_ARRAY_SQRT);


template <typename ValueType>
template <typename IndexType>
Array<ValueType> Array<ValueType>::distribute(
    std::shared_ptr<gko::Executor> exec,
    const IndexSet<IndexType> &index_set) const
{
    GKO_ASSERT_CONDITION(index_set.get_num_subsets() >= 1);
    using itype = IndexType;
    auto mpi_exec = as<gko::MpiExecutor>(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();

    itype num_subsets = index_set.get_num_subsets();
    auto num_subsets_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&num_subsets, 1, num_subsets_array.get_data(), 1,
                     root_rank);
    auto total_num_subsets =
        std::accumulate(num_subsets_array.get_data(),
                        num_subsets_array.get_data() + num_ranks, 0);
    mpi_exec->broadcast(&total_num_subsets, 1, root_rank);

    itype num_elems = static_cast<itype>(index_set.get_num_elems());
    auto num_elems_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&num_elems, 1, num_elems_array.get_data(), 1, root_rank);

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
    auto recv_count_array = gko::Array<int>{sub_exec->get_master(),
                                            num_subsets_array.get_num_elems()};
    detail::convert_data(
        sub_exec->get_master(), num_subsets_array.get_num_elems(),
        num_subsets_array.get_const_data(), recv_count_array.get_data());
    auto displ = gko::Array<int>{sub_exec->get_master(),
                                 num_subsets_array.get_num_elems()};
    detail::convert_data(sub_exec->get_master(),
                         num_subsets_array.get_num_elems(),
                         num_subsets_array.get_const_data(), displ.get_data());
    std::partial_sum(displ.get_data(), displ.get_data() + displ.get_num_elems(),
                     displ.get_data());
    for (auto i = 0; i < displ.get_num_elems(); ++i) {
        displ.get_data()[i] -= recv_count_array.get_data()[i];
    }
    mpi_exec->gather(start_idx_array.get_const_data(), int(num_subsets),
                     offset_array.get_data(), recv_count_array.get_const_data(),
                     displ.get_const_data(), root_rank);
    mpi_exec->gather(num_elems_in_subset.get_const_data(), int(num_subsets),
                     global_num_elems_subset_array.get_data(),
                     recv_count_array.get_const_data(), displ.get_const_data(),
                     root_rank);
    auto tag = gko::Array<itype>{sub_exec->get_master(),
                                 static_cast<size_type>(num_subsets)};
    for (auto t = 0; t < num_subsets; ++t) {
        tag.get_data()[t] = (my_rank + 1) * 1e4 + t;
    }
    auto tags = gko::Array<itype>{
        sub_exec->get_master(),
        static_cast<size_type>(num_ranks * total_num_subsets)};
    mpi_exec->gather(tag.get_const_data(), int(num_subsets), tags.get_data(),
                     recv_count_array.get_const_data(), displ.get_const_data(),
                     root_rank);
    auto work_array = Array<ValueType>{};
    auto dist_array = Array<ValueType>{};
    if (exec->get_master() == exec->get_sub_executor()) {
        dist_array = Array<ValueType>{exec, index_set.get_num_elems()};
        if (my_rank == root_rank) {
            work_array = Array<ValueType>{exec->get_master(), std::move(*this)};
        } else {
            work_array = Array<ValueType>{exec->get_master()};
        }
    } else {
#if GKO_HAVE_CUDA_AWARE_MPI
        dist_array = Array<ValueType>{exec, index_set.get_num_elems()};
        if (my_rank == root_rank) {
            work_array = Array<ValueType>{exec, std::move(*this)};
        } else {
            work_array = Array<ValueType>{exec};
        }
#else
        dist_array =
            Array<ValueType>{exec->get_master(), index_set.get_num_elems()};
        if (my_rank == root_rank) {
            work_array = Array<ValueType>{exec->get_master(), *this};
        } else {
            work_array = Array<ValueType>{exec->get_master()};
        }
#endif
    }

    auto req_array =
        mpi_exec->create_requests_array(num_ranks * total_num_subsets);
    auto idx = 0;
    for (auto in_rank = 0; in_rank < num_ranks; ++in_rank) {
        auto n_subsets = num_subsets_array.get_data()[in_rank];
        if (in_rank != root_rank) {
            if (my_rank == root_rank) {
                for (auto in_subset = 0; in_subset < n_subsets; ++in_subset) {
                    auto offset = offset_array.get_data()[idx];
                    auto g_n_elems =
                        global_num_elems_subset_array.get_data()[idx];
                    mpi_exec->send(&(work_array.get_const_data()[offset]),
                                   g_n_elems, in_rank, tags.get_data()[idx]);
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
            mpi_exec->recv(&dist_array.get_data()[offset], n_elems, root_rank,
                           tag.get_data()[in_subset]);
        } else {
            dist_array.get_executor()->get_mem_space()->copy_from(
                work_array.get_executor()->get_mem_space().get(), n_elems,
                &(work_array.get_const_data()[start_idx]),
                &dist_array.get_data()[offset]);
        }
        offset += n_elems;
    }
    if (exec->get_master() == exec->get_sub_executor()) {
        return std::move(dist_array);
    } else {
#if GKO_HAVE_CUDA_AWARE_MPI
        return std::move(dist_array);
#else
        auto dist_array_device =
            Array<ValueType>{exec, dist_array.get_num_elems()};
        dist_array_device = dist_array;
        return std::move(dist_array_device);
#endif
    }
}


#define GKO_DECLARE_ARRAY_DISTRIBUTE(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::distribute(         \
        std::shared_ptr<gko::Executor> exec,               \
        const IndexSet<IndexType> &index_set) const

GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_INDEX_TYPE(
    GKO_DECLARE_ARRAY_DISTRIBUTE);


template <typename ValueType>
template <typename IndexType>
Array<ValueType> Array<ValueType>::gather_on_root(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<IndexType> &index_set) const
{
    GKO_ASSERT_CONDITION(index_set.get_num_subsets() >= 1);
    using itype = IndexType;
    auto mpi_exec = as<gko::MpiExecutor>(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();

    itype num_subsets = index_set.get_num_subsets();
    auto num_subsets_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&num_subsets, 1, num_subsets_array.get_data(), 1,
                     root_rank);
    auto total_num_subsets =
        std::accumulate(num_subsets_array.get_data(),
                        num_subsets_array.get_data() + num_ranks, 0);
    mpi_exec->broadcast(&total_num_subsets, 1, root_rank);
    itype max_row_num = index_set.get_largest_element_in_set();
    auto max_row_num_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&max_row_num, 1, max_row_num_array.get_data(), 1,
                     root_rank);
    auto gathered_num_rows =
        (*std::max_element(max_row_num_array.get_data(),
                           max_row_num_array.get_data() + num_ranks)) +
        1;
    mpi_exec->broadcast(&gathered_num_rows, 1, root_rank);

    itype num_elems = static_cast<itype>(index_set.get_num_elems());
    auto num_elems_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&num_elems, 1, num_elems_array.get_data(), 1, root_rank);

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
    auto recv_count_array = gko::Array<int>{sub_exec->get_master(),
                                            num_subsets_array.get_num_elems()};
    detail::convert_data(
        sub_exec->get_master(), num_subsets_array.get_num_elems(),
        num_subsets_array.get_const_data(), recv_count_array.get_data());
    auto displ = gko::Array<int>{sub_exec->get_master(),
                                 num_subsets_array.get_num_elems()};
    detail::convert_data(sub_exec->get_master(),
                         num_subsets_array.get_num_elems(),
                         num_subsets_array.get_const_data(), displ.get_data());
    std::partial_sum(displ.get_data(), displ.get_data() + displ.get_num_elems(),
                     displ.get_data());
    for (auto i = 0; i < displ.get_num_elems(); ++i) {
        displ.get_data()[i] -= recv_count_array.get_data()[i];
    }
    mpi_exec->gather(start_idx_array.get_const_data(), int(num_subsets),
                     offset_array.get_data(), recv_count_array.get_const_data(),
                     displ.get_const_data(), root_rank);
    mpi_exec->gather(num_elems_in_subset.get_const_data(), int(num_subsets),
                     global_num_elems_subset_array.get_data(),
                     recv_count_array.get_const_data(), displ.get_const_data(),
                     root_rank);

    auto tag = gko::Array<itype>{sub_exec->get_master(),
                                 static_cast<size_type>(num_subsets)};
    for (auto t = 0; t < num_subsets; ++t) {
        tag.get_data()[t] = (my_rank + 1) * 1e4 + t;
    }
    auto tags = gko::Array<itype>{
        sub_exec->get_master(),
        static_cast<size_type>(num_ranks * total_num_subsets)};
    mpi_exec->gather(tag.get_const_data(), int(num_subsets), tags.get_data(),
                     recv_count_array.get_const_data(), displ.get_const_data(),
                     root_rank);

    auto work_array = Array<ValueType>{};
    auto gathered_array = Array<ValueType>{};
    if (exec->get_master() == exec->get_sub_executor()) {
        work_array = Array<ValueType>{exec, std::move(*this)};
        if (my_rank == root_rank) {
            gathered_array =
                Array<ValueType>{exec, size_type(gathered_num_rows)};
        }
    } else {
#if GKO_HAVE_CUDA_AWARE_MPI
        work_array = Array<ValueType>{exec, std::move(*this)};
        if (my_rank == root_rank) {
            gathered_array =
                Array<ValueType>{exec, size_type(gathered_num_rows)};
        }
#else
        work_array = Array<ValueType>{exec->get_master(), *this};
        if (my_rank == root_rank) {
            gathered_array = Array<ValueType>{exec->get_master(),
                                              size_type(gathered_num_rows)};
        }
#endif
    }

    auto offset = 0;
    for (auto in_subset = 0; in_subset < num_subsets; ++in_subset) {
        auto n_elems = num_elems_in_subset.get_data()[in_subset];
        auto start_idx = start_idx_array.get_data()[in_subset];
        if (my_rank != root_rank) {
            mpi_exec->send(&(work_array.get_const_data()[offset]), n_elems,
                           root_rank, tag.get_data()[in_subset]);
        } else {
            gathered_array.get_executor()->get_mem_space()->copy_from(
                work_array.get_executor()->get_mem_space().get(), n_elems,
                &(work_array.get_const_data()[offset]),
                &gathered_array.get_data()[start_idx]);
        }
        offset += n_elems;
    }

    auto idx = 0;
    for (auto in_rank = 0; in_rank < num_ranks; ++in_rank) {
        auto n_subsets = num_subsets_array.get_data()[in_rank];
        if (in_rank != root_rank) {
            if (my_rank == root_rank) {
                for (auto in_subset = 0; in_subset < n_subsets; ++in_subset) {
                    auto offset = offset_array.get_data()[idx];
                    auto g_n_elems =
                        global_num_elems_subset_array.get_data()[idx];
                    mpi_exec->recv(&gathered_array.get_data()[offset],
                                   g_n_elems, in_rank, tags.get_data()[idx]);
                    idx++;
                }
            }
        } else {
            idx += n_subsets;
        }
    }
    if (exec->get_master() == exec->get_sub_executor()) {
        return std::move(gathered_array);
    } else {
#if GKO_HAVE_CUDA_AWARE_MPI
        return std::move(gathered_array);
#else
        auto gathered_array_device =
            Array<ValueType>{exec, gathered_array.get_num_elems()};
        gathered_array_device = gathered_array;
        return std::move(gathered_array_device);
#endif
    }
}


#define GKO_DECLARE_ARRAY_GATHER_ON_ROOT(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::gather_on_root(         \
        std::shared_ptr<const gko::Executor> exec,             \
        const IndexSet<IndexType> &index_set) const

GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_INDEX_TYPE(
    GKO_DECLARE_ARRAY_GATHER_ON_ROOT);


template <typename ValueType>
template <typename IndexType>
Array<ValueType> Array<ValueType>::gather_on_all(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<IndexType> &index_set) const
{
    GKO_ASSERT_CONDITION(index_set.get_num_subsets() >= 1);
    using itype = IndexType;
    auto mpi_exec = as<gko::MpiExecutor>(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();

    itype num_subsets = index_set.get_num_subsets();
    auto num_subsets_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&num_subsets, 1, num_subsets_array.get_data(), 1,
                     root_rank);
    auto total_num_subsets =
        std::accumulate(num_subsets_array.get_data(),
                        num_subsets_array.get_data() + num_ranks, 0);
    mpi_exec->broadcast(&total_num_subsets, 1, root_rank);
    itype max_row_num = index_set.get_largest_element_in_set();
    auto max_row_num_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&max_row_num, 1, max_row_num_array.get_data(), 1,
                     root_rank);
    auto gathered_num_rows =
        (*std::max_element(max_row_num_array.get_data(),
                           max_row_num_array.get_data() + num_ranks)) +
        1;
    mpi_exec->broadcast(&gathered_num_rows, 1, root_rank);

    itype num_elems = static_cast<itype>(index_set.get_num_elems());
    auto num_elems_array =
        Array<itype>{sub_exec->get_master(), static_cast<size_type>(num_ranks)};
    mpi_exec->gather(&num_elems, 1, num_elems_array.get_data(), 1, root_rank);

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
    auto recv_count_array = gko::Array<int>{sub_exec->get_master(),
                                            num_subsets_array.get_num_elems()};
    detail::convert_data(
        sub_exec->get_master(), num_subsets_array.get_num_elems(),
        num_subsets_array.get_const_data(), recv_count_array.get_data());
    auto displ = gko::Array<int>{sub_exec->get_master(),
                                 num_subsets_array.get_num_elems()};
    detail::convert_data(sub_exec->get_master(),
                         num_subsets_array.get_num_elems(),
                         num_subsets_array.get_const_data(), displ.get_data());
    std::partial_sum(displ.get_data(), displ.get_data() + displ.get_num_elems(),
                     displ.get_data());
    for (auto i = 0; i < displ.get_num_elems(); ++i) {
        displ.get_data()[i] -= recv_count_array.get_data()[i];
    }
    mpi_exec->gather(start_idx_array.get_const_data(), int(num_subsets),
                     offset_array.get_data(), recv_count_array.get_const_data(),
                     displ.get_const_data(), root_rank);
    mpi_exec->gather(num_elems_in_subset.get_const_data(), int(num_subsets),
                     global_num_elems_subset_array.get_data(),
                     recv_count_array.get_const_data(), displ.get_const_data(),
                     root_rank);

    auto tag = gko::Array<itype>{sub_exec->get_master(),
                                 static_cast<size_type>(num_subsets)};
    for (auto t = 0; t < num_subsets; ++t) {
        tag.get_data()[t] = (my_rank + 1) * 1e4 + t;
    }
    auto tags = gko::Array<itype>{
        sub_exec->get_master(),
        static_cast<size_type>(num_ranks * total_num_subsets)};
    mpi_exec->gather(tag.get_const_data(), int(num_subsets), tags.get_data(),
                     recv_count_array.get_const_data(), displ.get_const_data(),
                     root_rank);


    auto work_array = Array<ValueType>{};
    auto gathered_array = Array<ValueType>{};
    if (exec->get_master() == exec->get_sub_executor()) {
        gathered_array = Array<ValueType>{exec, size_type(gathered_num_rows)};
        work_array = Array<ValueType>{exec->get_master(), std::move(*this)};
    } else {
#if GKO_HAVE_CUDA_AWARE_MPI
        gathered_array = Array<ValueType>{exec, size_type(gathered_num_rows)};
        work_array = Array<ValueType>{exec, std::move(*this)};
#else
        gathered_array =
            Array<ValueType>{exec->get_master(), size_type(gathered_num_rows)};
        work_array = Array<ValueType>{exec->get_master(), *this};
#endif
    }


    auto offset = 0;
    for (auto in_subset = 0; in_subset < num_subsets; ++in_subset) {
        auto n_elems = num_elems_in_subset.get_data()[in_subset];
        auto start_idx = start_idx_array.get_data()[in_subset];
        if (my_rank != root_rank) {
            mpi_exec->send(&(work_array.get_const_data()[offset]), n_elems,
                           root_rank, tag.get_data()[in_subset]);
        } else {
            gathered_array.get_executor()->get_mem_space()->copy_from(
                work_array.get_executor()->get_mem_space().get(), n_elems,
                &(work_array.get_const_data()[offset]),
                &gathered_array.get_data()[start_idx]);
        }
        offset += n_elems;
    }

    auto idx = 0;
    for (auto in_rank = 0; in_rank < num_ranks; ++in_rank) {
        auto n_subsets = num_subsets_array.get_data()[in_rank];
        if (in_rank != root_rank) {
            if (my_rank == root_rank) {
                for (auto in_subset = 0; in_subset < n_subsets; ++in_subset) {
                    auto offset = offset_array.get_data()[idx];
                    auto g_n_elems =
                        global_num_elems_subset_array.get_data()[idx];
                    mpi_exec->recv(&gathered_array.get_data()[offset],
                                   g_n_elems, in_rank, tags.get_data()[idx]);
                    idx++;
                }
            }
        } else {
            idx += n_subsets;
        }
    }
    mpi_exec->broadcast(gathered_array.get_data(),
                        gathered_array.get_num_elems(), root_rank);

    if (exec->get_master() == exec->get_sub_executor()) {
        return std::move(gathered_array);
    } else {
#if GKO_HAVE_CUDA_AWARE_MPI
        return std::move(gathered_array);
#else
        auto gathered_array_device =
            Array<ValueType>{exec, gathered_array.get_num_elems()};
        gathered_array_device = gathered_array;
        return std::move(gathered_array_device);
#endif
    }
}


#define GKO_DECLARE_ARRAY_GATHER_ON_ALL(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::gather_on_all(         \
        std::shared_ptr<const gko::Executor> exec,            \
        const IndexSet<IndexType> &index_set) const

GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_INDEX_TYPE(
    GKO_DECLARE_ARRAY_GATHER_ON_ALL);


template <typename ValueType>
template <typename IndexType>
Array<ValueType> Array<ValueType>::reduce_on_root(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<IndexType> &index_set,
    mpi::op_type op_enum) const GKO_NOT_IMPLEMENTED;

#define GKO_DECLARE_ARRAY_REDUCE_ON_ROOT(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::reduce_on_root(         \
        std::shared_ptr<const gko::Executor> exec,             \
        const IndexSet<IndexType> &index_set, mpi::op_type op_enum) const

GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_INDEX_TYPE(
    GKO_DECLARE_ARRAY_REDUCE_ON_ROOT);


template <typename ValueType>
template <typename IndexType>
Array<ValueType> Array<ValueType>::reduce_on_all(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<IndexType> &index_set,
    mpi::op_type op_enum) const GKO_NOT_IMPLEMENTED;

#define GKO_DECLARE_ARRAY_REDUCE_ON_ALL(ValueType, IndexType) \
    Array<ValueType> Array<ValueType>::reduce_on_all(         \
        std::shared_ptr<const gko::Executor> exec,            \
        const IndexSet<IndexType> &index_set, mpi::op_type op_enum) const

GKO_INSTANTIATE_FOR_EACH_VALUE_INDEX_AND_INDEX_TYPE(
    GKO_DECLARE_ARRAY_REDUCE_ON_ALL);


}  // namespace gko
