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

#include <ginkgo/core/matrix/dense.hpp>


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace matrix {
namespace dense {


GKO_REGISTER_OPERATION(simple_apply, dense::simple_apply);
GKO_REGISTER_OPERATION(apply, dense::apply);
GKO_REGISTER_OPERATION(scale, dense::scale);
GKO_REGISTER_OPERATION(add_scaled, dense::add_scaled);
GKO_REGISTER_OPERATION(add_scaled_diag, dense::add_scaled_diag);
GKO_REGISTER_OPERATION(compute_dot, dense::compute_dot);
GKO_REGISTER_OPERATION(compute_norm2, dense::compute_norm2);
GKO_REGISTER_OPERATION(count_nonzeros, dense::count_nonzeros);
GKO_REGISTER_OPERATION(calculate_max_nnz_per_row,
                       dense::calculate_max_nnz_per_row);
GKO_REGISTER_OPERATION(calculate_nonzeros_per_row,
                       dense::calculate_nonzeros_per_row);
GKO_REGISTER_OPERATION(calculate_total_cols, dense::calculate_total_cols);
GKO_REGISTER_OPERATION(transpose, dense::transpose);
GKO_REGISTER_OPERATION(conj_transpose, dense::conj_transpose);
GKO_REGISTER_OPERATION(row_permute, dense::row_permute);
GKO_REGISTER_OPERATION(column_permute, dense::column_permute);
GKO_REGISTER_OPERATION(inverse_row_permute, dense::inverse_row_permute);
GKO_REGISTER_OPERATION(inverse_column_permute, dense::inverse_column_permute);
GKO_REGISTER_OPERATION(convert_to_coo, dense::convert_to_coo);
GKO_REGISTER_OPERATION(convert_to_csr, dense::convert_to_csr);
GKO_REGISTER_OPERATION(convert_to_ell, dense::convert_to_ell);
GKO_REGISTER_OPERATION(convert_to_hybrid, dense::convert_to_hybrid);
GKO_REGISTER_OPERATION(convert_to_sellp, dense::convert_to_sellp);
GKO_REGISTER_OPERATION(convert_to_sparsity_csr, dense::convert_to_sparsity_csr);
GKO_REGISTER_OPERATION(extract_diagonal, dense::extract_diagonal);


}  // namespace dense


namespace {


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Coo<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(dense::make_count_nonzeros(source, &num_stored_nonzeros));
    auto tmp = Coo<ValueType, IndexType>::create(exec, source->get_size(),
                                                 num_stored_nonzeros);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Csr<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(dense::make_count_nonzeros(source, &num_stored_nonzeros));
    auto tmp = Csr<ValueType, IndexType>::create(
        exec, source->get_size(), num_stored_nonzeros, result->get_strategy());
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Ell<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();
    size_type num_stored_elements_per_row = 0;
    exec->run(dense::make_calculate_max_nnz_per_row(
        source, &num_stored_elements_per_row));
    const auto max_nnz_per_row = std::max(
        result->get_num_stored_elements_per_row(), num_stored_elements_per_row);
    const auto stride = std::max(result->get_stride(), source->get_size()[0]);
    auto tmp = Ell<ValueType, IndexType>::create(exec, source->get_size(),
                                                 max_nnz_per_row, stride);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Hybrid<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();
    Array<size_type> row_nnz(exec, source->get_size()[0]);
    exec->run(dense::make_calculate_nonzeros_per_row(source, &row_nnz));
    size_type ell_lim = zero<size_type>();
    size_type coo_lim = zero<size_type>();
    result->get_strategy()->compute_hybrid_config(row_nnz, &ell_lim, &coo_lim);
    const auto max_nnz_per_row =
        std::max(result->get_ell_num_stored_elements_per_row(), ell_lim);
    const auto stride =
        std::max(result->get_ell_stride(), source->get_size()[0]);
    const auto coo_nnz =
        std::max(result->get_coo_num_stored_elements(), coo_lim);
    auto tmp = Hybrid<ValueType, IndexType>::create(
        exec, source->get_size(), max_nnz_per_row, stride, coo_nnz,
        result->get_strategy());
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(Sellp<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();
    const auto stride_factor = (result->get_stride_factor() == 0)
                                   ? default_stride_factor
                                   : result->get_stride_factor();
    const auto slice_size = (result->get_slice_size() == 0)
                                ? default_slice_size
                                : result->get_slice_size();
    size_type total_cols = 0;
    exec->run(dense::make_calculate_total_cols(source, &total_cols,
                                               stride_factor, slice_size));
    auto tmp = Sellp<ValueType, IndexType>::create(
        exec, source->get_size(), slice_size, stride_factor, total_cols);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


template <typename ValueType, typename IndexType, typename MatrixType,
          typename OperationType>
inline void conversion_helper(SparsityCsr<ValueType, IndexType> *result,
                              MatrixType *source, const OperationType &op)
{
    auto exec = source->get_executor();

    size_type num_stored_nonzeros = 0;
    exec->run(dense::make_count_nonzeros(source, &num_stored_nonzeros));
    auto tmp = SparsityCsr<ValueType, IndexType>::create(
        exec, source->get_size(), num_stored_nonzeros);
    exec->run(op(source, tmp.get()));
    tmp->move_to(result);
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    auto exec = this->get_executor()->get_sub_executor();
    exec->run(dense::make_simple_apply(this, as<Dense<ValueType>>(b),
                                       as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *beta, LinOp *x) const
{
    auto exec = this->get_executor()->get_sub_executor();
    exec->run(dense::make_apply(
        as<Dense<ValueType>>(alpha), this, as<Dense<ValueType>>(b),
        as<Dense<ValueType>>(beta), as<Dense<ValueType>>(x)));
}


template <typename ValueType>
void Dense<ValueType>::distributed_apply_impl(const LinOp *b, LinOp *x) const
{
    auto mat_exec = this->get_executor()->get_sub_executor();
    auto b_exec = b->get_executor();
    auto dense_x = as<Dense<ValueType>>(x);
    auto dense_b = as<Dense<ValueType>>(b);

    auto flag = dense_b->get_size() == dense_b->get_global_size();
    if (!flag) {
        auto row_set_b = dense_b->get_index_set();
        auto gathered_rhs = dense_b->gather_on_all(b_exec, row_set_b);
        mat_exec->run(
            dense::make_simple_apply(this, gathered_rhs.get(), dense_x));
    } else {
        mat_exec->run(dense::make_simple_apply(this, dense_b, dense_x));
    }
}


template <typename ValueType>
void Dense<ValueType>::distributed_apply_impl(const LinOp *alpha,
                                              const LinOp *b, const LinOp *beta,
                                              LinOp *x) const
{
    auto exec = this->get_executor()->get_sub_executor();
    auto b_exec = b->get_executor();
    auto dense_x = as<Dense<ValueType>>(x);
    auto dense_b = as<Dense<ValueType>>(b);
    auto dense_alpha = as<Dense<ValueType>>(alpha);
    auto dense_beta = as<Dense<ValueType>>(beta);
    auto row_set_b = dense_b->get_index_set();
    auto flag = dense_b->get_size() == dense_b->get_global_size();
    if (!flag) {
        auto gathered_rhs = dense_b->gather_on_all(b_exec, row_set_b);
        exec->run(dense::make_apply(dense_alpha, this, gathered_rhs.get(),
                                    dense_beta, dense_x));
    } else {
        exec->run(
            dense::make_apply(dense_alpha, this, dense_b, dense_beta, dense_x));
    }
}


template <typename ValueType>
void Dense<ValueType>::scale_impl(const LinOp *alpha)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    auto exec = this->get_executor()->get_sub_executor();
    exec->run(dense::make_scale(as<Dense<ValueType>>(alpha), this));
}


template <typename ValueType>
void Dense<ValueType>::add_scaled_impl(const LinOp *alpha, const LinOp *b)
{
    GKO_ASSERT_EQUAL_ROWS(alpha, dim<2>(1, 1));
    if (alpha->get_size()[1] != 1) {
        // different alpha for each column
        GKO_ASSERT_EQUAL_COLS(this, alpha);
    }
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    auto exec = this->get_executor()->get_sub_executor();

    if (dynamic_cast<const Diagonal<ValueType> *>(b)) {
        exec->run(dense::make_add_scaled_diag(
            as<Dense<ValueType>>(alpha),
            dynamic_cast<const Diagonal<ValueType> *>(b), this));
        return;
    }

    exec->run(dense::make_add_scaled(as<Dense<ValueType>>(alpha),
                                     as<Dense<ValueType>>(b), this));
}


template <typename ValueType>
void Dense<ValueType>::compute_dot_impl(const LinOp *b, LinOp *result) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(this, b);
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto dense_vec = as<Dense<ValueType>>(this);
    auto dense_result = as<Dense<ValueType>>(result);
    auto dense_b = as<Dense<ValueType>>(b);
    exec->get_sub_executor()->run(
        dense::make_compute_dot(dense_vec, dense_b, dense_result));
    if (dynamic_cast<const gko::MpiExecutor *>(exec.get())) {
        auto mpi_exec = const_cast<gko::MpiExecutor *>(
            dynamic_cast<const gko::MpiExecutor *>(exec.get()));
        auto dense_res_exec = dense_result->get_executor();
        auto dense_res_host =
            Dense<ValueType>::create(dense_res_exec->get_master());
        if (exec->get_master() == exec->get_sub_executor()) {
            for (auto i = 0; i < this->get_size()[1]; ++i) {
                mpi_exec->all_reduce<ValueType>(&dense_result->get_values()[i],
                                                &dense_result->get_values()[i],
                                                1, mpi::op_type::sum);
            }
        } else {
#if GKO_HAVE_CUDA_AWARE_MPI
            for (auto i = 0; i < this->get_size()[1]; ++i) {
                mpi_exec->all_reduce<ValueType>(&dense_result->get_values()[i],
                                                &dense_result->get_values()[i],
                                                1, mpi::op_type::sum);
            }
#else
            dense_res_host->copy_from(dense_result);
            for (auto i = 0; i < this->get_size()[1]; ++i) {
                mpi_exec->all_reduce<ValueType>(
                    &dense_res_host->get_values()[i],
                    &dense_res_host->get_values()[i], 1, mpi::op_type::sum);
            }
            dense_result->copy_from(dense_res_host.get());
#endif
        }
    }
}


template <typename ValueType>
void Dense<ValueType>::compute_norm2_impl(LinOp *result) const
{
    using NonComplexType = remove_complex<ValueType>;
    using NormVector = Dense<NonComplexType>;
    using DenseVector = Dense<ValueType>;
    auto result_size = this->get_size();
    GKO_ASSERT_EQUAL_DIMENSIONS(result, dim<2>(1, this->get_size()[1]));
    auto exec = this->get_executor();
    auto sub_exec = exec->get_sub_executor();
    auto norm = as<NormVector>(result);
    if (dynamic_cast<const gko::MpiExecutor *>(exec.get())) {
        auto tmp_norm = DenseVector::create(sub_exec, result_size);
        sub_exec->run(dense::make_compute_dot(
            as<DenseVector>(this), as<DenseVector>(this), tmp_norm.get()));
        auto mpi_exec = const_cast<gko::MpiExecutor *>(
            dynamic_cast<const gko::MpiExecutor *>(exec.get()));
        auto norm_arr = gko::Array<NonComplexType>(
            sub_exec, gko::Array<NonComplexType>::view(sub_exec, result_size[1],
                                                       norm->get_values()));

        auto tmp_norm_host = Dense<ValueType>::create(sub_exec->get_master());
        if (exec->get_master() == sub_exec) {
            for (auto i = 0; i < result_size[1]; ++i) {
                mpi_exec->all_reduce<ValueType>(&tmp_norm->get_values()[i],
                                                &tmp_norm->get_values()[i], 1,
                                                mpi::op_type::sum);
            }
        } else {
#if GKO_HAVE_CUDA_AWARE_MPI
            for (auto i = 0; i < result_size[1]; ++i) {
                mpi_exec->all_reduce<ValueType>(&tmp_norm->get_values()[i],
                                                &tmp_norm->get_values()[i], 1,
                                                mpi::op_type::sum);
            }
#else
            tmp_norm_host->copy_from(tmp_norm.get());
            for (auto i = 0; i < result_size[1]; ++i) {
                mpi_exec->all_reduce<ValueType>(&tmp_norm_host->get_values()[i],
                                                &tmp_norm_host->get_values()[i],
                                                1, mpi::op_type::sum);
            }
            tmp_norm->copy_from(tmp_norm_host.get());
#endif
        }
        auto squared_norm = gko::Array<ValueType>(
            sub_exec, gko::Array<ValueType>::view(sub_exec, result_size[1],
                                                  tmp_norm->get_values()));
        squared_norm.sqrt(norm_arr);
    } else {
        sub_exec->run(
            dense::make_compute_norm2(as<Dense<ValueType>>(this), norm));
    }
}


template <typename ValueType>
void Dense<ValueType>::convert_to(
    Dense<next_precision<ValueType>> *result) const
{
    result->values_ = this->values_;
    result->stride_ = this->stride_;
    result->set_size(this->get_size());
    result->set_global_size(this->get_global_size());
}


template <typename ValueType>
void Dense<ValueType>::move_to(Dense<next_precision<ValueType>> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_coo<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Coo<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_coo<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Coo<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_csr<const Dense<ValueType> *&,
                                            decltype(result)>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Csr<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_csr<const Dense<ValueType> *&,
                                            decltype(result)>);
    result->make_srow();
}


template <typename ValueType>
void Dense<ValueType>::move_to(Csr<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_ell<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Ell<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_ell<const Dense<ValueType> *&,
                                            decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Ell<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_hybrid<const Dense<ValueType> *&,
                                               decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Hybrid<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_hybrid<const Dense<ValueType> *&,
                                               decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Hybrid<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sellp<const Dense<ValueType> *&,
                                              decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(Sellp<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sellp<const Dense<ValueType> *&,
                                              decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(Sellp<ValueType, int64> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(SparsityCsr<ValueType, int32> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sparsity_csr<const Dense<ValueType> *&,
                                                     decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(SparsityCsr<ValueType, int32> *result)
{
    this->convert_to(result);
}


template <typename ValueType>
void Dense<ValueType>::convert_to(SparsityCsr<ValueType, int64> *result) const
{
    conversion_helper(
        result, this,
        dense::template make_convert_to_sparsity_csr<const Dense<ValueType> *&,
                                                     decltype(result)>);
}


template <typename ValueType>
void Dense<ValueType>::move_to(SparsityCsr<ValueType, int64> *result)
{
    this->convert_to(result);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType *mtx, const MatrixData &data)
{
    auto tmp = MatrixType::create(mtx->get_executor()->get_master(), data.size);
    size_type ind = 0;
    for (size_type row = 0; row < data.size[0]; ++row) {
        for (size_type col = 0; col < data.size[1]; ++col) {
            if (ind < data.nonzeros.size() && data.nonzeros[ind].row == row &&
                data.nonzeros[ind].column == col) {
                tmp->at(row, col) = data.nonzeros[ind].value;
                ++ind;
            } else {
                tmp->at(row, col) = zero<typename MatrixType::value_type>();
            }
        }
    }
    tmp->move_to(mtx);
}


template <typename MatrixType, typename MatrixData>
inline void read_impl(MatrixType *mtx, const MatrixData &data,
                      const Array<size_type> &dist)
{
    auto exec = mtx->get_executor();
    GKO_ASSERT_MPI_EXEC(exec.get());
    auto mpi_exec = as<gko::MpiExecutor>(exec.get());
    auto rank = mpi_exec->get_my_rank();
    auto local_num_rows = dist.get_num_elems();
    auto row_idx_set = gko::IndexSet<size_type>{data.size[0]};
    row_idx_set.add_indices(dist.get_const_data(),
                            dist.get_const_data() + local_num_rows);
    auto tmp = MatrixType::create(exec->get_master(), data.size, row_idx_set,
                                  data.size[1]);
    auto lnnz_max = local_num_rows * data.size[1];
    size_type lnnz = 0;
    size_type col = 0;
    for (size_type nnz = 0; nnz < data.nonzeros.size(); ++nnz) {
        if (lnnz < lnnz_max && row_idx_set.is_element(data.nonzeros[nnz].row)) {
            if (data.nonzeros[nnz].column == col) {
                tmp->get_values()[lnnz] = data.nonzeros[nnz].value;
            } else {
                tmp->get_values()[lnnz] =
                    zero<typename MatrixType::value_type>();
            }
            if (col < data.size[1] - 1) {
                col++;
            } else {
                col = 0;
            }
            ++lnnz;
        }
    }
    tmp->move_to(mtx);
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::read(const mat_data &data)
{
    read_impl(this, data);
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data32 &data)
{
    read_impl(this, data);
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data &data, const Array<size_type> &dist)
{
    read_impl(this, data, dist);
}


template <typename ValueType>
void Dense<ValueType>::read(const mat_data32 &data,
                            const Array<size_type> &dist)
{
    read_impl(this, data, dist);
}


namespace {


template <typename MatrixType, typename MatrixData>
inline void write_impl(const MatrixType *mtx, MatrixData &data)
{
    std::unique_ptr<const LinOp> op{};
    const MatrixType *tmp{};
    if (mtx->get_executor()->get_master() != mtx->get_executor()) {
        op = mtx->clone(mtx->get_executor()->get_master());
        tmp = static_cast<const MatrixType *>(op.get());
    } else {
        tmp = mtx;
    }

    data = {mtx->get_size(), {}};

    for (size_type row = 0; row < data.size[0]; ++row) {
        for (size_type col = 0; col < data.size[1]; ++col) {
            if (tmp->at(row, col) != zero<typename MatrixType::value_type>()) {
                data.nonzeros.emplace_back(row, col, tmp->at(row, col));
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void Dense<ValueType>::write(mat_data &data) const
{
    write_impl(this, data);
}


template <typename ValueType>
void Dense<ValueType>::write(mat_data32 &data) const
{
    write_impl(this, data);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Dense::create(exec, gko::transpose(this->get_size()));

    exec->run(dense::make_transpose(this, trans_cpy.get()));

    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::conj_transpose() const
{
    auto exec = this->get_executor();
    auto trans_cpy = Dense::create(exec, gko::transpose(this->get_size()));

    exec->run(dense::make_conj_transpose(this, trans_cpy.get()));
    return std::move(trans_cpy);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::gather_on_root(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<size_type> &row_set) const
{
    GKO_ASSERT_MPI_EXEC(exec.get());
    auto mpi_exec = gko::as<MpiExecutor>(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();

    auto mat_size = this->get_size();
    auto mat_stride = this->get_stride();
    auto local_num_rows = row_set.get_num_elems();
    GKO_ASSERT_CONDITION(mat_size[0] == local_num_rows);
    auto global_num_rows = local_num_rows;
    mpi_exec->all_reduce(&local_num_rows, &global_num_rows, 1,
                         gko::mpi::op_type::sum);
    auto max_index_size = row_set.get_largest_element_in_set();
    auto index_set =
        gko::IndexSet<gko::int32>{(max_index_size + 1) * mat_stride};
    auto elem = row_set.begin();
    for (auto i = 0; i < local_num_rows; ++i) {
        index_set.add_dense_row(*elem, mat_stride);
        elem++;
    }
    auto gathered_array =
        this->get_const_values_array().gather_on_root(exec, index_set);
    auto gathered_dense = Dense::create(exec);
    if (my_rank == root_rank) {
        gathered_dense =
            Dense::create(exec, gko::dim<2>(global_num_rows, mat_size[1]),
                          row_set, gathered_array, mat_stride);
    }
    return std::move(gathered_dense);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::gather_on_all(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<size_type> &row_set) const
{
    GKO_ASSERT_MPI_EXEC(exec.get());
    auto mpi_exec = gko::as<MpiExecutor>(exec.get());
    auto sub_exec = exec->get_sub_executor();
    auto num_ranks = mpi_exec->get_num_ranks();
    auto my_rank = mpi_exec->get_my_rank();
    auto root_rank = mpi_exec->get_root_rank();

    auto mat_size = this->get_size();
    auto mat_stride = this->get_stride();
    auto local_num_rows = row_set.get_num_elems();
    GKO_ASSERT_CONDITION(mat_size[0] == local_num_rows);
    auto global_num_rows = local_num_rows;
    mpi_exec->all_reduce(&local_num_rows, &global_num_rows, 1,
                         gko::mpi::op_type::sum);
    auto max_index_size = row_set.get_largest_element_in_set();
    auto index_set =
        gko::IndexSet<gko::int32>{(max_index_size + 1) * mat_stride};
    auto elem = row_set.begin();
    for (auto i = 0; i < local_num_rows; ++i) {
        index_set.add_dense_row(*elem, mat_stride);
        elem++;
    }
    auto gathered_array =
        this->get_const_values_array().gather_on_all(exec, index_set);
    auto gathered_dense =
        Dense::create(exec, gko::dim<2>(global_num_rows, mat_size[1]), row_set,
                      gathered_array, mat_stride);
    return std::move(gathered_dense);
}


template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::reduce_on_root(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<size_type> &row_set,
    mpi::op_type op_enum) const GKO_NOT_IMPLEMENTED;

template <typename ValueType>
std::unique_ptr<Dense<ValueType>> Dense<ValueType>::reduce_on_all(
    std::shared_ptr<const gko::Executor> exec,
    const IndexSet<size_type> &row_set,
    mpi::op_type op_enum) const GKO_NOT_IMPLEMENTED;


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(
        dense::make_row_permute(permutation_indices, this, permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const Array<int32> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_column_permute(permutation_indices, this,
                                         permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::row_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[0]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(
        dense::make_row_permute(permutation_indices, this, permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::column_permute(
    const Array<int64> *permutation_indices) const
{
    GKO_ASSERT_EQ(permutation_indices->get_num_elems(), this->get_size()[1]);
    auto exec = this->get_executor();
    auto permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_column_permute(permutation_indices, this,
                                         permute_cpy.get()));

    return std::move(permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const Array<int32> *inverse_permutation_indices) const
{
    GKO_ASSERT_EQ(inverse_permutation_indices->get_num_elems(),
                  this->get_size()[0]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_row_permute(inverse_permutation_indices, this,
                                              inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const Array<int32> *inverse_permutation_indices) const
{
    GKO_ASSERT_EQ(inverse_permutation_indices->get_num_elems(),
                  this->get_size()[1]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_column_permute(
        inverse_permutation_indices, this, inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_row_permute(
    const Array<int64> *inverse_permutation_indices) const
{
    GKO_ASSERT_EQ(inverse_permutation_indices->get_num_elems(),
                  this->get_size()[0]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_row_permute(inverse_permutation_indices, this,
                                              inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<LinOp> Dense<ValueType>::inverse_column_permute(
    const Array<int64> *inverse_permutation_indices) const
{
    GKO_ASSERT_EQ(inverse_permutation_indices->get_num_elems(),
                  this->get_size()[1]);
    auto exec = this->get_executor();
    auto inverse_permute_cpy = Dense::create(exec, this->get_size());

    exec->run(dense::make_inverse_column_permute(
        inverse_permutation_indices, this, inverse_permute_cpy.get()));

    return std::move(inverse_permute_cpy);
}


template <typename ValueType>
std::unique_ptr<Diagonal<ValueType>> Dense<ValueType>::extract_diagonal() const
{
    auto exec = this->get_executor();

    const auto diag_size = std::min(this->get_size()[0], this->get_size()[1]);
    auto diag = Diagonal<ValueType>::create(exec, diag_size);
    exec->run(dense::make_extract_diagonal(this, lend(diag)));
    return diag;
}


#define GKO_DECLARE_DENSE_MATRIX(_type) class Dense<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_MATRIX);


}  // namespace matrix


}  // namespace gko
