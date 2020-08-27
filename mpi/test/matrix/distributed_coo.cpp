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
#include <ginkgo/core/matrix/coo.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename ValueIndexType>
class DistributedCoo : public ::testing::Test {
protected:
    using value_type =
        typename std::tuple_element<0, decltype(ValueIndexType())>::type;
    using index_type =
        typename std::tuple_element<1, decltype(ValueIndexType())>::type;
    using Mtx = gko::matrix::Coo<value_type, index_type>;
    DistributedCoo() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::ReferenceExecutor::create();
        mpi_exec = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
        sub_exec = mpi_exec->get_sub_executor();
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

    void assert_empty(const Mtx *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
        ASSERT_EQ(m->get_const_values(), nullptr);
        ASSERT_EQ(m->get_const_col_idxs(), nullptr);
        ASSERT_EQ(m->get_const_row_idxs(), nullptr);
    }

    static void assert_equal_mtxs(
        const gko::matrix::Coo<value_type, index_type> *m,
        const gko::matrix::Coo<value_type, index_type> *lm)
    {
        ASSERT_EQ(m->get_size(), lm->get_size());
        ASSERT_EQ(m->get_num_stored_elements(), lm->get_num_stored_elements());

        for (auto i = 0; i < m->get_num_stored_elements(); ++i) {
            EXPECT_EQ(m->get_const_values()[i], lm->get_const_values()[i]);
            EXPECT_EQ(m->get_const_col_idxs()[i], lm->get_const_col_idxs()[i]);
            EXPECT_EQ(m->get_const_row_idxs()[i], lm->get_const_row_idxs()[i]);
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
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const gko::Executor> sub_exec;
    int rank;
};

// TODO: Some weird error with double and complex types for COO apply
TYPED_TEST_CASE(DistributedCoo, gko::test::DebugValueIndexType);
// TYPED_TEST_CASE(DistributedCoo, gko::test::ValueIndexTypes);


TYPED_TEST(DistributedCoo, DoesNotThrowForMpiExecutor)
{
    using Mtx = typename TestFixture::Mtx;
    ASSERT_NO_THROW(Mtx::distributed_create(this->mpi_exec));
}

TYPED_TEST(DistributedCoo, ThrowsForOtherExecutors)
{
    using Mtx = typename TestFixture::Mtx;
    ASSERT_THROW(Mtx::distributed_create(this->exec), gko::NotSupported);
}

TYPED_TEST(DistributedCoo, CanBeEmpty)
{
    using Mtx = typename TestFixture::Mtx;
    auto empty = Mtx::distributed_create(this->mpi_exec);
    this->assert_empty(empty.get());
}


TYPED_TEST(DistributedCoo, CanBeConstructedFromExistingExecutorData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    value_type *values;
    index_type *col_idxs;
    index_type *row_idxs;
    if (this->rank == 0) {
        values = new value_type[4]{1.0, 2.0, 3.0, 4.0};
        col_idxs = new index_type[4]{0, 1, 1, 0};
        row_idxs = new index_type[4]{0, 0, 1, 1};
    } else {
        values = new value_type[4]{1.0, 2.5, 3.0, 4.0};
        col_idxs = new index_type[4]{0, 0, 1, 0};
        row_idxs = new index_type[4]{0, 1, 0, 1};
    }

    auto mtx = gko::matrix::Coo<value_type, index_type>::distributed_create(
        this->mpi_exec, gko::dim<2>{3, 2},
        gko::Array<value_type>::view(this->sub_exec, 4, values),
        gko::Array<index_type>::view(this->sub_exec, 4, col_idxs),
        gko::Array<index_type>::view(this->sub_exec, 4, row_idxs));

    ASSERT_EQ(mtx->get_global_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(mtx->get_size(), gko::dim<2>(3, 2));
    ASSERT_EQ(mtx->get_const_values(), values);
    ASSERT_EQ(mtx->get_const_col_idxs(), col_idxs);
    ASSERT_EQ(mtx->get_const_row_idxs(), row_idxs);
    delete values, col_idxs, row_idxs;
}


TYPED_TEST(DistributedCoo, CanDistributeData)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    gko::dim<2> global_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_idxs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_idxs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
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
        row_idxs =
            new index_type[16]{0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4};
        local_values = new value_type[5]{1.0, 1.0, -1.0, 2.0, 1.5};
        local_col_idxs = new index_type[5]{0, 2, 4, 1, 4};
        local_row_idxs = new index_type[5]{0, 0, 0, 1, 1};
        row_dist.add_subset(0, 2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         *  0.0  2.0  0.0  0.0  1.5 ] rank 0
         */
        // clang-format on
        l_mat = Mtx::create(
            this->sub_exec, local_size,
            gko::Array<value_type>::view(this->sub_exec, 5, local_values),
            gko::Array<index_type>::view(this->sub_exec, 5, local_col_idxs),
            gko::Array<index_type>::view(this->sub_exec, 5, local_row_idxs));
    } else {
        local_values = new value_type[11]{-2.0, 4.0, 6.0,  0.5, -2.0, 3.0,
                                          5.0,  1.0, -3.0, 4.0, 7.0};
        local_col_idxs = new index_type[11]{0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_idxs = new index_type[11]{2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4};
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
        l_mat = Mtx::create(
            this->sub_exec, local_size,
            gko::Array<value_type>::view(this->sub_exec, 11, local_values),
            gko::Array<index_type>::view(this->sub_exec, 11, local_col_idxs),
            gko::Array<index_type>::view(this->sub_exec, 11, local_row_idxs));
    }
    global_size = gko::dim<2>(5, 5);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec, 16, values),
        gko::Array<index_type>::view(this->sub_exec, 16, col_idxs),
        gko::Array<index_type>::view(this->sub_exec, 16, row_idxs));

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    this->assert_equal_mtxs(mat.get(), l_mat.get());
    if (this->rank == 0) {
        delete values, row_idxs, col_idxs;
    }
    delete local_values, local_row_idxs, local_col_idxs;
}


TYPED_TEST(DistributedCoo, CanDistributeDataNonContiguously)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    gko::dim<2> global_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_idxs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_idxs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    if (this->rank == 0) {
        // clang-format off
        /*  1.0  0.0  1.0  0.0 -1.0 ] rank 0
         *  0.0  2.0  0.0  0.0  1.5 ] rank 1
         * -2.0  0.0  4.0  0.0  6.0 } rank 1
         *  0.5 -2.0  3.0  5.0  1.0 } rank 0
         * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        values = new value_type[16]{1.0, 1.0,  -1.0, 2.0, 1.5, -2.0, 4.0, 6.0,
                                    0.5, -2.0, 3.0,  5.0, 1.0, -3.0, 4.0, 7.0};
        col_idxs =
            new index_type[16]{0, 2, 4, 1, 4, 0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        row_idxs =
            new index_type[16]{0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4};
        local_values =
            new value_type[8]{1.0, 1.0, -1.0, 0.5, -2.0, 3.0, 5.0, 1.0};
        local_col_idxs = new index_type[8]{0, 2, 4, 0, 1, 2, 3, 4};
        local_row_idxs = new index_type[8]{0, 0, 0, 3, 3, 3, 3, 3};
        row_dist.add_index(0);
        row_dist.add_index(3);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         *  0.5 -2.0  3.0  5.0  1.0 } rank 0
         */
        // clang-format on
        l_mat = Mtx::create(
            this->sub_exec, local_size,
            gko::Array<value_type>::view(this->sub_exec, 8, local_values),
            gko::Array<index_type>::view(this->sub_exec, 8, local_col_idxs),
            gko::Array<index_type>::view(this->sub_exec, 8, local_row_idxs));
    } else {
        local_values =
            new value_type[8]{2.0, 1.5, -2.0, 4.0, 6.0, -3.0, 4.0, 7.0};
        local_col_idxs = new index_type[8]{1, 4, 0, 2, 4, 0, 1, 4};
        local_row_idxs = new index_type[8]{1, 1, 2, 2, 2, 4, 4, 4};
        row_dist.add_index(1);
        row_dist.add_index(2);
        row_dist.add_index(4);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        // clang-format off
         /*  0.0  2.0  0.0  0.0  1.5 ] rank 1
          * -2.0  0.0  4.0  0.0  6.0 } rank 1
          * -3.0  4.0  0.0  0.0  7.0 } rank 1
         */
        // clang-format on
        l_mat = Mtx::create(
            this->sub_exec, local_size,
            gko::Array<value_type>::view(this->sub_exec, 8, local_values),
            gko::Array<index_type>::view(this->sub_exec, 8, local_col_idxs),
            gko::Array<index_type>::view(this->sub_exec, 8, local_row_idxs));
    }
    global_size = gko::dim<2>(5, 5);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec, 16, values),
        gko::Array<index_type>::view(this->sub_exec, 16, col_idxs),
        gko::Array<index_type>::view(this->sub_exec, 16, row_idxs));

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    this->assert_equal_mtxs(mat.get(), l_mat.get());
    if (this->rank == 0) {
        delete values, row_idxs, col_idxs;
    }
    delete local_values, local_row_idxs, local_col_idxs;
}


TYPED_TEST(DistributedCoo, AppliesToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using index_type = typename TestFixture::index_type;
    using size_type = gko::size_type;
    using DenseVec = gko::matrix::Dense<value_type>;
    std::shared_ptr<Mtx> mat{};
    std::shared_ptr<Mtx> l_mat{};
    std::shared_ptr<DenseVec> dvec{};
    std::shared_ptr<DenseVec> expected{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    gko::dim<2> global_size{};
    value_type *values;
    index_type *col_idxs;
    index_type *row_idxs;
    value_type *local_values;
    index_type *local_col_idxs;
    index_type *local_row_idxs;
    gko::IndexSet<size_type> row_dist{6};
    size_type num_rows;
    value_type *vec_data;
    vec_data = new value_type[5]{-3.0, 3.0, -5.0, 5.0, 1.0};
    dvec = DenseVec::create(
        this->sub_exec, gko::dim<2>(5, 1),
        gko::Array<value_type>::view(this->sub_exec, 5, vec_data), 1);
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
        row_idxs =
            new index_type[16]{0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4};
        local_values = new value_type[5]{1.0, 1.0, -1.0, 2.0, 1.5};
        local_col_idxs = new index_type[5]{0, 2, 4, 1, 4};
        local_row_idxs = new index_type[5]{0, 0, 0, 1, 1};
        row_dist.add_subset(0, 2);
        num_rows = 2;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec, gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         *  1.0  0.0  1.0  0.0 -1.0 ]
         *  0.0  2.0  0.0  0.0  1.5 ] rank 0
         */
        // clang-format on
        l_mat = Mtx::create(
            this->sub_exec, local_size,
            gko::Array<value_type>::view(this->sub_exec, 5, local_values),
            gko::Array<index_type>::view(this->sub_exec, 5, local_col_idxs),
            gko::Array<index_type>::view(this->sub_exec, 5, local_row_idxs));
    } else {
        local_values = new value_type[11]{-2.0, 4.0, 6.0,  0.5, -2.0, 3.0,
                                          5.0,  1.0, -3.0, 4.0, 7.0};
        local_col_idxs = new index_type[11]{0, 2, 4, 0, 1, 2, 3, 4, 0, 1, 4};
        local_row_idxs = new index_type[11]{2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4};
        row_dist.add_subset(2, 5);
        num_rows = 3;
        local_size = gko::dim<2>(num_rows, 5);
        expected = DenseVec::create(this->sub_exec, gko::dim<2>(num_rows, 1));
        // clang-format off
        /*
         * -2.0  0.0  4.0  0.0  6.0 }
         *  0.5 -2.0  3.0  5.0  1.0 } rank 1
         * -3.0  4.0  0.0  0.0  7.0 }
         */
        // clang-format on
        l_mat = Mtx::create(
            this->sub_exec, local_size,
            gko::Array<value_type>::view(this->sub_exec, 11, local_values),
            gko::Array<index_type>::view(this->sub_exec, 11, local_col_idxs),
            gko::Array<index_type>::view(this->sub_exec, 11, local_row_idxs));
    }
    global_size = gko::dim<2>(5, 5);
    mat = Mtx::create_and_distribute(
        this->mpi_exec, global_size, row_dist,
        gko::Array<value_type>::view(this->sub_exec, 16, values),
        gko::Array<index_type>::view(this->sub_exec, 16, col_idxs),
        gko::Array<index_type>::view(this->sub_exec, 16, row_idxs));
    std::shared_ptr<DenseVec> res =
        DenseVec::create(this->sub_exec, gko::dim<2>(num_rows, 1));
    l_mat->apply(dvec.get(), expected.get());
    mat->apply(dvec.get(), res.get());

    ASSERT_EQ(mat->get_executor(), this->mpi_exec);
    this->assert_equal_mtxs(mat.get(), l_mat.get());
    this->assert_equal_vecs(res.get(), expected.get());
    if (this->rank == 0) {
        delete values, row_idxs, col_idxs;
    }
    delete vec_data, local_values, local_row_idxs, local_col_idxs;
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;