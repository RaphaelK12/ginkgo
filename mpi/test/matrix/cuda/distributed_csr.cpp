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


#include <mpi.h>

#include <gtest/gtest.h>

#include "gtest-mpi-listener.hpp"
#include "gtest-mpi-main.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class DistributedCsr : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Csr<value_type, index_type>;
    DistributedCsr() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::ReferenceExecutor::create();
        host = gko::ReferenceExecutor::create();
        GKO_ASSERT_NO_MPI_ERRORS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        mpi_exec = gko::MpiExecutor::create(gko::CudaExecutor::create(0, host));
        mpi_exec2 = gko::MpiExecutor::create(host);
        sub_exec = mpi_exec->get_sub_executor();
        sub_exec2 = mpi_exec2->get_sub_executor();
        rank = mpi_exec->get_my_rank();
        ASSERT_GT(mpi_exec->get_num_ranks(), 1);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    static void assert_equal_mtxs(
        const gko::matrix::Csr<value_type, index_type> *m,
        const gko::matrix::Csr<value_type, index_type> *lm)
    {
        ASSERT_EQ(m->get_size(), lm->get_size());
        EXPECT_EQ(m->get_num_stored_elements(), lm->get_num_stored_elements());

        for (auto i = 0; i < lm->get_size()[0] + 1; ++i) {
            EXPECT_EQ(m->get_const_row_ptrs()[i], lm->get_const_row_ptrs()[i]);
        }

        for (auto i = 0; i < lm->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m->get_const_values()[i], lm->get_const_values()[i]);
            EXPECT_EQ(m->get_const_col_idxs()[i], lm->get_const_col_idxs()[i]);
        }
    }

    static void assert_equal_vecs(const gko::matrix::Dense<value_type> *m,
                                  const gko::matrix::Dense<value_type> *lm)
    {
        ASSERT_EQ(m->get_size(), lm->get_size());
        ASSERT_EQ(m->get_stride(), lm->get_stride());
        ASSERT_EQ(m->get_num_stored_elements(), lm->get_num_stored_elements());

        for (auto i = 0; i < m->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m->get_const_values()[i], lm->get_const_values()[i]);
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<gko::MpiExecutor> mpi_exec2;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::Executor> host;
    std::shared_ptr<const gko::Executor> sub_exec;
    std::shared_ptr<const gko::Executor> sub_exec2;
    int rank;
};

TYPED_TEST_CASE(DistributedCsr, gko::test::ValueIndexTypes);


TYPED_TEST(DistributedCsr, CanBeConstructedFromExistingExecutorData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    if (this->rank == 0) {
        values = new value_type[4]{1.0, 2.0, 3.0, 4.0};
        col_idxs = new index_type[4]{0, 1, 1, 0};
        row_ptrs = new index_type[4]{0, 2, 3, 4};
    } else {
        values = new value_type[4]{1.0, 2.5, 3.0, 4.0};
        col_idxs = new index_type[4]{0, 0, 1, 0};
        row_ptrs = new index_type[4]{0, 1, 3, 4};
    }

    auto mtx = gko::matrix::Csr<value_type, index_type>::distributed_create(
        this->mpi_exec, gko::dim<2>{3, 2},
        gko::Array<value_type>::view(this->sub_exec->get_master(), 4, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 4, col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 4, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));
    auto mtx_h = Mtx::create(this->mpi_exec2);
    mtx_h->copy_from(mtx.get());
    auto mtx2 = gko::matrix::Csr<value_type, index_type>::distributed_create(
        this->mpi_exec2, gko::dim<2>{3, 2},
        gko::Array<value_type>::view(this->sub_exec2->get_master(), 4, values),
        gko::Array<index_type>::view(this->sub_exec2->get_master(), 4,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec2->get_master(), 4,
                                     row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));

    ASSERT_EQ(mtx->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(mtx->get_global_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(mtx->get_num_srow_elements(), 1);
    this->assert_equal_mtxs(mtx_h.get(), mtx2.get());
    delete values, col_idxs, row_ptrs;
}


TYPED_TEST(DistributedCsr, CanDistributeData)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_ptrs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    auto global_size = gko::dim<2>(5, 5);
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ]
         *  0.0  2.0  0.0  0.0  1.5 ] rank 0
         * -2.0  0.0  4.0  0.0  6.0 }
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 }
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_ptrs = new index_type[6]{0, 3, 5, 8, 13, 16};
        local_values = new value_type[5]{1.0, 1.0, -1.0, 2.0, 1.5};
        local_col_idxs = new index_type[5]{0, 2, 4, 1, 4};
        local_row_ptrs = new index_type[3]{0, 3, 5};
        row_dist.add_subset(0, 2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         *  0.0  2.0  0.0  0.0  1.5 ] rank 0
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), gko::dim<2>{2, 5},
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 5, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 5, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 3, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    } else {
        local_values = new value_type[11]{-2.0, 4.0, 6.0,  0.5, -2.0, 3.0,
                                          5.0,  1.0, -3.0, 4.0, 7.0};
        local_col_idxs = new index_type[11]{0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_ptrs = new index_type[4]{0, 3, 8, 11};
        row_dist.add_subset(2, 5);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
        /*
         * -2.0  0.0  4.0  0.0  6.0 }
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 }
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), local_size,
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 11, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 11, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 4, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    }
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec->get_master(), 16, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 16,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 6, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    auto mat_h = Mtx::create(this->mpi_exec2);
    mat_h->copy_from(mat.get());
    this->assert_equal_mtxs(mat_h.get(), l_mat.get());
    if (this->rank == 0) {
        delete values, row_ptrs, col_idxs;
    }
    delete local_values, local_row_ptrs, local_col_idxs;
}


TYPED_TEST(DistributedCsr, CanDistributeDataNonContiguously)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_ptrs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    auto global_size = gko::dim<2>(5, 5);
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ] rank 0
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_ptrs = new index_type[6]{0, 3, 5, 8, 13, 16};
        local_values = new value_type[6]{1.0, 1.0, -1.0, -2.0, 4.0, 6.0};
        local_col_idxs = new index_type[6]{0, 2, 4, 0, 2, 4};
        local_row_ptrs = new index_type[3]{0, 3, 6};
        row_dist.add_index(0);
        row_dist.add_index(2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), gko::dim<2>{2, 5},
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 6, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 6, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 3, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    } else {
        local_values = new value_type[10]{2.0, 1.5, 0.5,  -2.0, 3.0,
                                          5.0, 1.0, -3.0, 4.0,  7.0};
        local_col_idxs = new index_type[10]{1, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_ptrs = new index_type[4]{0, 2, 7, 10};
        row_dist.add_index(1);
        row_dist.add_index(3);
        row_dist.add_index(4);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
        /*
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), local_size,
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 10, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 10, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 4, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    }
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec->get_master(), 16, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 16,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 6, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    auto mat_h = Mtx::create(this->mpi_exec2);
    mat_h->copy_from(mat.get());
    this->assert_equal_mtxs(mat_h.get(), l_mat.get());
    if (this->rank == 0) {
        delete values, row_ptrs, col_idxs;
    }
    delete local_values, local_row_ptrs, local_col_idxs;
}


TYPED_TEST(DistributedCsr, AppliesToLocalDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using DenseVec = gko::matrix::Dense<value_type>;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    std::shared_ptr<DenseVec> dvec{};
    std::shared_ptr<DenseVec> expected{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_ptrs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    auto global_size = gko::dim<2>(5, 5);
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ] rank 0
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_ptrs = new index_type[6]{0, 3, 5, 8, 13, 16};
        local_values = new value_type[6]{1.0, 1.0, -1.0, -2.0, 4.0, 6.0};
        local_col_idxs = new index_type[6]{0, 2, 4, 0, 2, 4};
        local_row_ptrs = new index_type[3]{0, 3, 6};
        row_dist.add_index(0);
        row_dist.add_index(2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), gko::dim<2>{2, 5},
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 6, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 6, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 3, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    } else {
        local_values = new value_type[10]{2.0, 1.5, 0.5,  -2.0, 3.0,
                                          5.0, 1.0, -3.0, 4.0,  7.0};
        local_col_idxs = new index_type[10]{1, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_ptrs = new index_type[4]{0, 2, 7, 10};
        row_dist.add_index(1);
        row_dist.add_index(3);
        row_dist.add_index(4);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), local_size,
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 10, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 10, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 4, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    }
    auto rhs = gko::initialize<DenseVec>(
        1, I<value_type>({-3.0, 3.0, -5.0, 5.0, 1.0}), this->sub_exec);
    auto exact = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({-9.0, 7.5, -8.0, 3.5, 28.0}),
        this->mpi_exec2);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec->get_master(), 16, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 16,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 6, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));
    std::shared_ptr<DenseVec> res =
        DenseVec::create(this->sub_exec, gko::dim<2>(num_rows, 1));
    l_mat->apply(rhs.get(), expected.get());
    mat->apply(rhs.get(), res.get());

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    auto mat_h = Mtx::create(this->mpi_exec2);
    mat_h->copy_from(mat.get());
    this->assert_equal_mtxs(mat_h.get(), l_mat.get());
    auto res_h = DenseVec::create(this->mpi_exec2);
    res_h->copy_from(res.get());
    this->assert_equal_vecs(res_h.get(), exact.get());
    if (this->rank == 0) {
        delete values, row_ptrs, col_idxs;
    }
    delete local_values, local_row_ptrs, local_col_idxs;
}


TYPED_TEST(DistributedCsr, AppliesToDistributedDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using DenseVec = gko::matrix::Dense<value_type>;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    std::shared_ptr<DenseVec> dvec{};
    std::shared_ptr<DenseVec> expected{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_ptrs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    auto global_size = gko::dim<2>(5, 5);
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ] rank 0
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_ptrs = new index_type[6]{0, 3, 5, 8, 13, 16};
        local_values = new value_type[6]{1.0, 1.0, -1.0, -2.0, 4.0, 6.0};
        local_col_idxs = new index_type[6]{0, 2, 4, 0, 2, 4};
        local_row_ptrs = new index_type[3]{0, 3, 6};
        row_dist.add_index(0);
        row_dist.add_index(2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), gko::dim<2>{2, 5},
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 6, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 6, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 3, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    } else {
        local_values = new value_type[10]{2.0, 1.5, 0.5,  -2.0, 3.0,
                                          5.0, 1.0, -3.0, 4.0,  7.0};
        local_col_idxs = new index_type[10]{1, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_ptrs = new index_type[4]{0, 2, 7, 10};
        row_dist.add_index(1);
        row_dist.add_index(3);
        row_dist.add_index(4);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), local_size,
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 10, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 10, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 4, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    }
    auto rhs =
        gko::initialize<DenseVec>(1, I<value_type>({-3.0, 3.0, -5.0, 5.0, 1.0}),
                                  this->sub_exec->get_master());
    auto dist_rhs = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({-3.0, 3.0, -5.0, 5.0, 1.0}),
        this->mpi_exec);
    auto exact = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({-9.0, 7.5, -8.0, 3.5, 28.0}),
        this->mpi_exec2);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec->get_master(), 16, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 16,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 6, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));
    std::shared_ptr<DenseVec> res =
        DenseVec::create(this->sub_exec, gko::dim<2>(num_rows, 1));
    l_mat->apply(rhs.get(), expected.get());
    mat->apply(dist_rhs.get(), res.get());
    ASSERT_EQ(mat->get_executor(), this->mpi_exec);

    auto mat_h = Mtx::create(this->mpi_exec2);
    mat_h->copy_from(mat.get());
    this->assert_equal_mtxs(mat_h.get(), l_mat.get());
    auto res_h = DenseVec::create(this->mpi_exec2);
    res_h->copy_from(res.get());
    this->assert_equal_vecs(res_h.get(), exact.get());
    if (this->rank == 0) {
        delete values, row_ptrs, col_idxs;
    }
    delete local_values, local_row_ptrs, local_col_idxs;
}


TYPED_TEST(DistributedCsr, AdvancedAppliesToLocalDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using DenseVec = gko::matrix::Dense<value_type>;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    std::shared_ptr<DenseVec> dvec{};
    std::shared_ptr<DenseVec> expected{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_ptrs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    auto global_size = gko::dim<2>(5, 5);
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ] rank 0
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_ptrs = new index_type[6]{0, 3, 5, 8, 13, 16};
        local_values = new value_type[6]{1.0, 1.0, -1.0, -2.0, 4.0, 6.0};
        local_col_idxs = new index_type[6]{0, 2, 4, 0, 2, 4};
        local_row_ptrs = new index_type[3]{0, 3, 6};
        row_dist.add_index(0);
        row_dist.add_index(2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), gko::dim<2>{2, 5},
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 6, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 6, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 3, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    } else {
        local_values = new value_type[10]{2.0, 1.5, 0.5,  -2.0, 3.0,
                                          5.0, 1.0, -3.0, 4.0,  7.0};
        local_col_idxs = new index_type[10]{1, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_ptrs = new index_type[4]{0, 2, 7, 10};
        row_dist.add_index(1);
        row_dist.add_index(3);
        row_dist.add_index(4);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), local_size,
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 10, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 10, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 4, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    }
    auto alpha = gko::initialize<DenseVec>({I<value_type>({-1.0})},
                                           this->sub_exec->get_master());
    auto beta = gko::initialize<DenseVec>({I<value_type>({2.0})},
                                          this->sub_exec->get_master());
    auto rhs = gko::initialize<DenseVec>(
        1, I<value_type>({-3.0, 3.0, -5.0, 5.0, 1.0}), this->sub_exec);
    auto exact = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({11.0, -6.5, 7.0, -5.5, -25.0}),
        this->mpi_exec2);
    auto res = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({1.0, 0.5, -0.5, -1.0, 1.5}),
        this->mpi_exec);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec->get_master(), 16, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 16,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 6, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));
    l_mat->apply(alpha.get(), rhs.get(), beta.get(), expected.get());
    mat->apply(alpha.get(), rhs.get(), beta.get(), res.get());
    ASSERT_EQ(mat->get_executor(), this->mpi_exec);

    auto mat_h = Mtx::create(this->mpi_exec2);
    mat_h->copy_from(mat.get());
    this->assert_equal_mtxs(mat_h.get(), l_mat.get());
    auto res_h = DenseVec::create(this->mpi_exec2);
    res_h->copy_from(res.get());
    this->assert_equal_vecs(res_h.get(), exact.get());
    if (this->rank == 0) {
        delete values, row_ptrs, col_idxs;
    }
    delete local_values, local_row_ptrs, local_col_idxs;
}


TYPED_TEST(DistributedCsr, AdvancedAppliesToDistributedDense)
{
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using Mtx = typename TestFixture::Mtx;
    using DenseVec = gko::matrix::Dense<value_type>;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    std::shared_ptr<DenseVec> dvec{};
    std::shared_ptr<DenseVec> expected{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_ptrs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_ptrs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    auto global_size = gko::dim<2>(5, 5);
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ] rank 0
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_ptrs = new index_type[6]{0, 3, 5, 8, 13, 16};
        local_values = new value_type[6]{1.0, 1.0, -1.0, -2.0, 4.0, 6.0};
        local_col_idxs = new index_type[6]{0, 2, 4, 0, 2, 4};
        local_row_ptrs = new index_type[3]{0, 3, 6};
        row_dist.add_index(0);
        row_dist.add_index(2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         * -2.0  0.0  4.0  0.0  6.0 } rank 0
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), gko::dim<2>{2, 5},
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 6, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 6, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 3, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    } else {
        local_values = new value_type[10]{2.0, 1.5, 0.5,  -2.0, 3.0,
                                          5.0, 1.0, -3.0, 4.0,  7.0};
        local_col_idxs = new index_type[10]{1, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_ptrs = new index_type[4]{0, 2, 7, 10};
        row_dist.add_index(1);
        row_dist.add_index(3);
        row_dist.add_index(4);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec->get_master(),
                                    gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        l_mat =
            Mtx::create(this->sub_exec->get_master(), local_size,
                        gko::Array<value_type>::view(
                            this->sub_exec->get_master(), 10, local_values),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 10, local_col_idxs),
                        gko::Array<index_type>::view(
                            this->sub_exec->get_master(), 4, local_row_ptrs),
                        std::make_shared<typename Mtx::load_balance>(2));
    }
    auto alpha =
        gko::initialize<DenseVec>({I<value_type>({-1.0})}, this->sub_exec);
    auto beta =
        gko::initialize<DenseVec>({I<value_type>({2.0})}, this->sub_exec);
    auto rhs =
        gko::initialize<DenseVec>(1, I<value_type>({-3.0, 3.0, -5.0, 5.0, 1.0}),
                                  this->sub_exec->get_master());
    auto dist_rhs = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({-3.0, 3.0, -5.0, 5.0, 1.0}),
        this->mpi_exec);
    auto exact = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({11.0, -6.5, 7.0, -5.5, -25.0}),
        this->mpi_exec2);
    auto res = gko::initialize_and_distribute<DenseVec>(
        1, row_dist, I<value_type>({1.0, 0.5, -0.5, -1.0, 1.5}),
        this->mpi_exec);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec->get_master(), 16, values),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 16,
                                     col_idxs),
        gko::Array<index_type>::view(this->sub_exec->get_master(), 6, row_ptrs),
        std::make_shared<typename Mtx::load_balance>(2));
    l_mat->apply(alpha.get(), rhs.get(), beta.get(), expected.get());
    mat->apply(alpha.get(), dist_rhs.get(), beta.get(), res.get());

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    auto mat_h = Mtx::create(this->mpi_exec2);
    mat_h->copy_from(mat.get());
    this->assert_equal_mtxs(mat_h.get(), l_mat.get());
    auto res_h = DenseVec::create(this->mpi_exec2);
    res_h->copy_from(res.get());
    this->assert_equal_vecs(res_h.get(), exact.get());
    if (this->rank == 0) {
        delete values, row_ptrs, col_idxs;
    }
    delete local_values, local_row_ptrs, local_col_idxs;
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
