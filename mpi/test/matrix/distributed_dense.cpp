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
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class DistributedDense : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    DistributedDense() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::ReferenceExecutor::create();
        mpi_exec = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
        sub_exec = mpi_exec->get_sub_executor();
        rank = mpi_exec->get_my_rank();
        ASSERT_GT(mpi_exec->get_num_ranks(), 1);
        mtx1 = gko::initialize<Mtx>({I<T>({1.0, -1.0}), I<T>({-2.0, 2.0})},
                                    sub_exec);
        mtx2 =
            gko::initialize<Mtx>({{1.0, 2.0, 3.0}, {0.5, 1.5, 2.5}}, sub_exec);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    static void assert_empty(gko::matrix::Dense<value_type> *m)
    {
        ASSERT_EQ(m->get_size(), gko::dim<2>(0, 0));
        ASSERT_EQ(m->get_num_stored_elements(), 0);
    }

    static void assert_equal_mtxs(const gko::matrix::Dense<value_type> *m,
                                  const gko::matrix::Dense<value_type> *lm)
    {
        EXPECT_EQ(m->get_size()[0], lm->get_size()[0]);
        EXPECT_EQ(m->get_size()[1], lm->get_size()[1]);
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
    std::unique_ptr<Mtx> mtx1;
    std::unique_ptr<Mtx> mtx2;
    int rank;
};

TYPED_TEST_CASE(DistributedDense, gko::test::ValueTypes);


TYPED_TEST(DistributedDense, DoesNotThrowForMpiExecutor)
{
    ASSERT_NO_THROW(
        gko::matrix::Dense<TypeParam>::distributed_create(this->mpi_exec));
}

TYPED_TEST(DistributedDense, ThrowsForOtherExecutors)
{
    ASSERT_THROW(gko::matrix::Dense<TypeParam>::distributed_create(this->exec),
                 gko::NotSupported);
}

TYPED_TEST(DistributedDense, CanBeEmpty)
{
    auto empty =
        gko::matrix::Dense<TypeParam>::distributed_create(this->mpi_exec);
    this->assert_empty(empty.get());
}

TYPED_TEST(DistributedDense, ReturnsNullValuesArrayWhenEmpty)
{
    auto empty =
        gko::matrix::Dense<TypeParam>::distributed_create(this->mpi_exec);
    ASSERT_EQ(empty->get_const_values(), nullptr);
}


TYPED_TEST(DistributedDense, CanBeConstructedWithSize)
{
    auto m = gko::matrix::Dense<TypeParam>::distributed_create(
        this->mpi_exec, gko::dim<2>{2, 3});

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 3);
    ASSERT_EQ(m->get_num_stored_elements(), 6);
}


TYPED_TEST(DistributedDense, CanBeConstructedWithSizeAndStride)
{
    auto m = gko::matrix::Dense<TypeParam>::distributed_create(
        this->mpi_exec, gko::dim<2>{2, 3}, 4);

    ASSERT_EQ(m->get_size(), gko::dim<2>(2, 3));
    EXPECT_EQ(m->get_stride(), 4);
    ASSERT_EQ(m->get_num_stored_elements(), 8);
}


TYPED_TEST(DistributedDense,
           ColumnVectorCanBeInitializedWithInitializeAndStride)
{
    using Mtx = gko::matrix::Dense<TypeParam>;
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> local_mtx{};
    std::shared_ptr<Mtx> dist_mtx{};
    gko::IndexSet<size_type> row_set{6};
    if (this->rank == 0) {
        local_mtx = gko::initialize<Mtx>(1, {2, -1.0, 0.0}, this->sub_exec);
        row_set.add_subset(0, 3);
    } else {
        local_mtx = gko::initialize<Mtx>(1, {-1.0, 2, -1.0}, this->sub_exec);
        row_set.add_subset(3, 6);
    }

    dist_mtx = gko::initialize_and_distribute<Mtx>(
        1, row_set, {2, -1.0, 0.0, -1.0, 2, -1.0}, this->mpi_exec);
    this->assert_equal_mtxs(local_mtx.get(), dist_mtx.get());
}


TYPED_TEST(DistributedDense, CanBeInitializedWithInitializeAndStride)
{
    using Mtx = gko::matrix::Dense<TypeParam>;
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> local_mtx{};
    std::shared_ptr<Mtx> dist_mtx{};
    gko::IndexSet<size_type> row_set{3};
    if (this->rank == 0) {
        local_mtx = gko::initialize<Mtx>(3, {{2, -1.0, 0.0}, {0.0, -1.0, 2}},
                                         this->sub_exec);
        row_set.add_index(0);
        row_set.add_index(2);
    } else {
        local_mtx = gko::initialize<Mtx>(3, {{-1.0, 2, -1.0}}, this->sub_exec);
        row_set.add_index(1);
    }

    dist_mtx = gko::initialize_and_distribute<Mtx>(
        3, row_set, {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}},
        this->mpi_exec);
    this->assert_equal_mtxs(local_mtx.get(), dist_mtx.get());
}


TYPED_TEST(DistributedDense, CanBeInitializedWithInitializeWithoutStride)
{
    using Mtx = gko::matrix::Dense<TypeParam>;
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    std::shared_ptr<Mtx> local_mtx{};
    std::shared_ptr<Mtx> dist_mtx{};
    gko::IndexSet<size_type> row_set{3};
    if (this->rank == 0) {
        local_mtx = gko::initialize<Mtx>({{2, -1.0, 0.0}, {0.0, -1.0, 2}},
                                         this->sub_exec);
        row_set.add_index(0);
        row_set.add_index(2);
    } else {
        local_mtx = gko::initialize<Mtx>({{-1.0, 2, -1.0}}, this->sub_exec);
        row_set.add_index(1);
    }

    dist_mtx = gko::initialize_and_distribute<Mtx>(
        row_set, {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}},
        this->mpi_exec);
    this->assert_equal_mtxs(local_mtx.get(), dist_mtx.get());
}


TYPED_TEST(DistributedDense, CanBeConstructedFromExistingExecutorData)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    if (this->rank == 0) {
        // clang-format off
      data = new value_type[9]{
                       1.0, 2.0, -1.0,
                       3.0, 4.0, -1.0,
                       5.0, 6.0, -1.0};
        // clang-format on
    } else {
        // clang-format off
      data = new value_type[9]{
                         1.0, 5.0, -1.0,
                         3.0, 2.0, -1.0,
                         5.0, 7.0, -1.0};
        // clang-format on
    }

    auto m = gko::matrix::Dense<TypeParam>::distributed_create(
        this->mpi_exec, gko::dim<2>{3, 2},
        gko::Array<value_type>::view(this->sub_exec, 9, data), 3);

    ASSERT_EQ(m->get_const_values(), data);
    if (this->rank == 0) {
        ASSERT_EQ(m->at(2, 1), value_type{6.0});
    } else {
        ASSERT_EQ(m->at(2, 1), value_type{7.0});
    }
    delete data;
}


TYPED_TEST(DistributedDense, CanDistributeDataUsingRowAndStride)
{
    using value_type = typename TestFixture::value_type;
    using index_type = gko::int32;
    using size_type = gko::size_type;
    value_type *data;
    value_type *comp_data;
    size_type num_rows;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    std::shared_ptr<gko::matrix::Dense<value_type>> m{};
    std::shared_ptr<gko::matrix::Dense<value_type>> lm{};
    this->mpi_exec->set_root_rank(0);
    gko::dim<2> local_size{};
    gko::IndexSet<size_type> row_set{6};
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[20]{
                                 1.0, 2.0, -1.0, 2.0,
                                 3.0, 4.0, -1.0, 3.0,
                                 3.0, 4.0, -1.0, 3.0,
                                 3.0, 4.0, -1.0, 3.0,
                                 5.0, 6.0, -1.0, 4.0};
        comp_data = new value_type[8]{
                                 1.0, 2.0, -1.0, 2.0,
                                 3.0, 4.0, -1.0, 3.0};
        // clang-format on
        row_set.add_subset(0, 2);
        local_size = gko::dim<2>(2, 4);
        lm = gko::matrix::Dense<value_type>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 8, comp_data), 4);
    } else {
        // clang-format off
        comp_data = new value_type[12]{3.0, 4.0, -1.0, 3.0,
                3.0, 4.0, -1.0, 3.0,
                5.0, 6.0, -1.0, 4.0};
        // clang-format on
        row_set.add_subset(2, 5);
        local_size = gko::dim<2>(3, 4);
        lm = gko::matrix::Dense<value_type>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 12, comp_data), 4);
    }
    num_rows = row_set.get_num_elems();
    m = gko::matrix::Dense<value_type>::create_and_distribute(
        this->mpi_exec, local_size, row_set,
        gko::Array<value_type>::view(sub_exec, 20, data), 4);

    ASSERT_EQ(m->get_executor(), this->mpi_exec);
    this->assert_equal_mtxs(m.get(), lm.get());
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedDense, CanDistributeDataNonContiguously)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    std::shared_ptr<gko::matrix::Dense<value_type>> m{};
    std::shared_ptr<gko::matrix::Dense<value_type>> lm{};
    gko::IndexSet<gko::int32> row_set{5};
    gko::dim<2> local_size{};
    this->mpi_exec->set_root_rank(0);
    auto sub_exec = this->mpi_exec->get_sub_executor();
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[20]{
                                 1.0, 2.0, -1.0, 2.0,
                                 3.0, 4.0, -1.0, 3.0,
                                 3.0, 2.0, 1.0, 3.0,
                                 3.0, 1.0, -1.0, 3.0,
                                 5.0, 6.0, -1.0, 4.0};
        comp_data = new value_type[8]{
                                 1.0, 2.0, -1.0, 2.0,
                                 5.0, 6.0, -1.0, 4.0};
        // clang-format on
        local_size = gko::dim<2>(2, 4);
        lm = gko::matrix::Dense<TypeParam>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 8, comp_data), 4);
        row_set.add_index(0);
        row_set.add_index(4);
    } else {
        // clang-format off
        comp_data = new value_type[12]{3.0, 4.0, -1.0, 3.0,
                                       3.0, 2.0, 1.0, 3.0,
                                       3.0, 1.0, -1.0, 3.0};
        // clang-format on
        row_set.add_subset(1, 4);
        local_size = gko::dim<2>(3, 4);
        lm = gko::matrix::Dense<TypeParam>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 12, comp_data), 4);
    }
    m = gko::matrix::Dense<value_type>::create_and_distribute(
        this->mpi_exec, local_size, row_set,
        gko::Array<value_type>::view(this->sub_exec, 20, data), 4);

    ASSERT_EQ(m->get_executor(), this->mpi_exec);
    this->assert_equal_mtxs(m.get(), lm.get());
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedDense, CanCollectNonContiguousDenseMatricesOnRoot)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    value_type *data;
    value_type *comp_data;
    std::shared_ptr<gko::matrix::Dense<value_type>> m{};
    std::shared_ptr<gko::matrix::Dense<value_type>> lm{};
    std::shared_ptr<gko::matrix::Dense<value_type>> comp_dense{};
    gko::IndexSet<size_type> row_set{6};
    gko::dim<2> local_size{};
    this->mpi_exec->set_root_rank(0);
    auto sub_exec = this->mpi_exec->get_sub_executor();
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[20]{
                                 1.0, 2.0, -1.0, 2.0,
                                 3.0, 4.0, -1.0, 3.0,
                                 3.0, 2.0, 1.0, 3.0,
                                 3.0, 1.0, -1.0, 3.0,
                                 5.0, 6.0, -1.0, 4.0};
        comp_data = new value_type[8]{
                                 1.0, 2.0, -1.0, 2.0,
                                 5.0, 6.0, -1.0, 4.0};
        // clang-format on
        comp_dense = gko::matrix::Dense<TypeParam>::create(
            sub_exec, gko::dim<2>(5, 4),
            gko::Array<value_type>::view(sub_exec, 20, data), 4);
        local_size = gko::dim<2>(2, 4);
        lm = gko::matrix::Dense<TypeParam>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 8, comp_data), 4);
        row_set.add_index(0);
        row_set.add_index(4);
    } else {
        // clang-format off
        comp_data = new value_type[12]{3.0, 4.0, -1.0, 3.0,
                                       3.0, 2.0, 1.0, 3.0,
                                       3.0, 1.0, -1.0, 3.0};
        // clang-format on
        row_set.add_subset(1, 4);
        local_size = gko::dim<2>(3, 4);
        lm = gko::matrix::Dense<TypeParam>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 12, comp_data), 4);
    }
    m = lm->collect_on_root(this->mpi_exec, row_set);

    if (this->rank == 0) {
        ASSERT_EQ(m->get_executor(), this->mpi_exec);
        this->assert_equal_mtxs(m.get(), comp_dense.get());
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedDense, CanCollectNonContiguousDenseMatricesOnAllRanks)
{
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    value_type *data;
    value_type *comp_data;
    std::shared_ptr<gko::matrix::Dense<value_type>> m{};
    std::shared_ptr<gko::matrix::Dense<value_type>> lm{};
    std::shared_ptr<gko::matrix::Dense<value_type>> comp_dense{};
    gko::IndexSet<size_type> row_set{6};
    gko::dim<2> local_size{};
    this->mpi_exec->set_root_rank(0);
    auto sub_exec = this->mpi_exec->get_sub_executor();
    if (this->rank == 0) {
        // clang-format off
        comp_data = new value_type[8]{
                                 1.0, 2.0, -1.0, 2.0,
                                 5.0, 6.0, -1.0, 4.0};
        // clang-format on
        local_size = gko::dim<2>(2, 4);
        lm = gko::matrix::Dense<TypeParam>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 8, comp_data), 4);
        row_set.add_index(0);
        row_set.add_index(4);
    } else {
        // clang-format off
        comp_data = new value_type[12]{3.0, 4.0, -1.0, 3.0,
                                       3.0, 2.0, 1.0, 3.0,
                                       3.0, 1.0, -1.0, 3.0};
        // clang-format on
        row_set.add_subset(1, 4);
        local_size = gko::dim<2>(3, 4);
        lm = gko::matrix::Dense<TypeParam>::create(
            sub_exec, local_size,
            gko::Array<value_type>::view(sub_exec, 12, comp_data), 4);
    }
    // clang-format off
    data = new value_type[20]{
                             1.0, 2.0, -1.0, 2.0,
                             3.0, 4.0, -1.0, 3.0,
                             3.0, 2.0, 1.0, 3.0,
                             3.0, 1.0, -1.0, 3.0,
                             5.0, 6.0, -1.0, 4.0};
    // clang-format on
    comp_dense = gko::matrix::Dense<TypeParam>::create(
        sub_exec, gko::dim<2>(5, 4),
        gko::Array<value_type>::view(sub_exec, 20, data), 4);
    m = lm->collect_on_all(this->mpi_exec, row_set);

    ASSERT_EQ(m->get_executor(), this->mpi_exec);
    this->assert_equal_mtxs(m.get(), comp_dense.get());
    delete data;
    delete comp_data;
}


TYPED_TEST(DistributedDense, AppliesToDense)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    gko::IndexSet<gko::int32> index_set{6};
    gko::dim<2> local_size{};
    gko::dim<2> res_size{};
    std::shared_ptr<gko::matrix::Dense<value_type>> comp_res;
    value_type *data;
    value_type *comp_data;
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[10]{ 1.0, 2.0,
                                  -1.0, 2.0,
                                   3.0, 4.0,
                                  -1.0, 3.0,
                                   3.0, 4.0};
        comp_data = new value_type[4]{ -3.0, 3.0,
                                        -5.0, 5.0};
        // clang-format on
        index_set.add_subset(0, 2);
        local_size = gko::dim<2>(2, 2);
        res_size = gko::dim<2>(2, 2);
        comp_res = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 4, comp_data), 2);
    } else {
        // clang-format off
        comp_data = new value_type[6]{-5.0, 5.0,
                                      -7.0, 7.0,
                                      -5.0, 5.0};
        // clang-format on
        index_set.add_subset(2, 5);
        local_size = gko::dim<2>(3, 2);
        res_size = gko::dim<2>(3, 2);
        comp_res = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 6, comp_data), 2);
    }
    auto mat = gko::matrix::Dense<value_type>::create_and_distribute(
        this->mpi_exec, local_size, index_set,
        gko::Array<value_type>::view(this->sub_exec, 10, data), 2);
    auto res = gko::matrix::Dense<value_type>::create(
        this->mpi_exec->get_sub_executor(), res_size);
    mat->apply(this->mtx1.get(), res.get());
    this->assert_equal_mtxs(res.get(), comp_res.get());
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedDense, ScalesDense)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    gko::IndexSet<gko::int32> index_set{6};
    gko::dim<2> local_size{};
    gko::dim<2> res_size{};
    std::shared_ptr<Mtx> comp_res;
    auto alpha = gko::initialize<Mtx>({I<value_type>{2.0, -2.0}},
                                      this->mpi_exec->get_sub_executor());
    value_type *data;
    value_type *comp_data;
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[10]{ 1.0, 2.0,
                                  -1.0, 2.0,
                                   3.0, 4.0,
                                  -1.0, 3.0,
                                   3.0, 4.0};
        comp_data = new value_type[4]{ 2.0, -4.0,
                                        -2.0, -4.0};
        // clang-format on
        index_set.add_subset(0, 2);
        local_size = gko::dim<2>(2, 2);
        res_size = gko::dim<2>(2, 2);
        comp_res = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 4, comp_data), 2);
    } else {
        // clang-format off
        comp_data = new value_type[6]{6.0, -8.0,
                                      -2.0, -6.0,
                                      6, -8.0};
        // clang-format on
        index_set.add_subset(2, 5);
        local_size = gko::dim<2>(3, 2);
        res_size = gko::dim<2>(3, 2);
        comp_res = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 6, comp_data), 2);
    }
    auto mat = gko::matrix::Dense<value_type>::create_and_distribute(
        this->mpi_exec, local_size, index_set,
        gko::Array<value_type>::view(this->sub_exec, 10, data), 2);
    mat->scale(alpha.get());
    this->assert_equal_mtxs(mat.get(), comp_res.get());
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(DistributedDense, AddsScaled)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    gko::IndexSet<gko::int32> index_set{6};
    gko::dim<2> local_size{};
    gko::dim<2> res_size{};
    std::shared_ptr<Mtx> comp_res;
    std::shared_ptr<Mtx> b;
    auto alpha = gko::initialize<Mtx>({I<value_type>{2.0, -2.0}},
                                      this->mpi_exec->get_sub_executor());
    value_type *data;
    value_type *b_data;
    value_type *comp_data;
    if (this->rank == 0) {
        // clang-format off
        data = new value_type[10]{ 1.0, 2.0,
                                  -1.0, 2.0,
                                   3.0, 4.0,
                                  -1.0, 3.0,
                                   3.0, 4.0};
        b_data = new value_type[4]{ 1.0, -2.0,
                                    0.0, -3.0};
        comp_data = new value_type[4]{ 3.0, 6.0,
                                      -1.0, 8.0};
        // clang-format on
        index_set.add_subset(0, 2);
        local_size = gko::dim<2>(2, 2);
        res_size = gko::dim<2>(2, 2);
        comp_res = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 4, comp_data), 2);
        b = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 4, b_data), 2);
    } else {
        // clang-format off
        b_data = new value_type[6]{ 1.0, -2.0,
                                    0.0, 3.0,
                                    0.5, -3.0};
        comp_data = new value_type[6]{5.0, 8.0,
                                     -1.0, -3.0,
                                      4.0, 10.0};
        // clang-format on
        index_set.add_subset(2, 5);
        local_size = gko::dim<2>(3, 2);
        res_size = gko::dim<2>(3, 2);
        comp_res = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 6, comp_data), 2);
        b = gko::matrix::Dense<value_type>::create(
            this->mpi_exec->get_sub_executor(), res_size,
            gko::Array<value_type>::view(this->sub_exec, 6, b_data), 2);
    }
    auto mat = gko::matrix::Dense<value_type>::create_and_distribute(
        this->mpi_exec, local_size, index_set,
        gko::Array<value_type>::view(this->sub_exec, 10, data), 2);
    mat->add_scaled(alpha.get(), b.get());
    this->assert_equal_mtxs(mat.get(), comp_res.get());
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
    delete b_data;
}


TYPED_TEST(DistributedDense, CanComputeDot)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    gko::dim<2> vec2_size{};
    gko::dim<2> vec1_size{};
    std::shared_ptr<Mtx> vec1;
    std::shared_ptr<Mtx> vec2;
    auto res1 = gko::initialize<Mtx>({I<value_type>{0.0, 0.0}},
                                     this->mpi_exec->get_sub_executor());
    auto res2 = gko::initialize<Mtx>({I<value_type>{0.0, 0.0}},
                                     this->mpi_exec->get_sub_executor());
    auto comp_res = gko::initialize<Mtx>({I<value_type>{30.0, 20.0}},
                                         this->mpi_exec->get_sub_executor());
    value_type *vec1_data;
    value_type *vec2_data;
    if (this->rank == 0) {
        // clang-format off
        vec1_data = new value_type[6]{ 1.0, 1.0,
                -2.0, 0.0,
                6.0, 2.0};
        vec2_data = new value_type[6]{-1.0, 2.0,
                1.0, 2.0,
                3.0, 4.0};
        // clang-format on
    } else {
        // clang-format off
        vec1_data = new value_type[6]{ 1.0, 1.0,
                                       -2.0, 0.0,
                                        6.0, 2.0};
        vec2_data = new value_type[6]{-1.0, 2.0,
                1.0, 2.0,
                3.0, 4.0};
        // clang-format on
    }
    vec1_size = gko::dim<2>(3, 2);
    vec2_size = gko::dim<2>(3, 2);
    vec1 = gko::matrix::Dense<value_type>::distributed_create(
        this->mpi_exec, vec1_size,
        gko::Array<value_type>::view(this->sub_exec, 6, vec1_data), 2);
    vec2 = gko::matrix::Dense<value_type>::distributed_create(
        this->mpi_exec, vec2_size,
        gko::Array<value_type>::view(this->sub_exec, 6, vec2_data), 2);
    vec1->compute_dot(vec2.get(), res1.get());
    vec2->compute_dot(vec1.get(), res2.get());
    this->assert_equal_mtxs(comp_res.get(), res1.get());
    this->assert_equal_mtxs(comp_res.get(), res2.get());
    delete vec1_data;
    delete vec2_data;
}


TYPED_TEST(DistributedDense, CanCompute2Norm)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using T_nc = gko::remove_complex<value_type>;
    using NormVector = gko::matrix::Dense<T_nc>;
    gko::dim<2> vec2_size{};
    gko::dim<2> vec1_size{};
    std::shared_ptr<Mtx> vec1;
    std::shared_ptr<Mtx> vec2;
    auto res1 = gko::initialize<NormVector>({I<T_nc>{0.0, 0.0}},
                                            this->mpi_exec->get_sub_executor());
    gko::size_type nelems;
    value_type *vec1_data;
    if (this->rank == 0) {
        // clang-format off
        vec1_data = new value_type[6]{ 1.0, 1.0,
                -2.0, 0.0,
                6.0, 2.0};
        // clang-format on
        vec1_size = gko::dim<2>(3, 2);
        nelems = 6;
    } else {
        // clang-format off
        vec1_data = new value_type[4]{ 1.5, -0.5,
                                        5.0, 4.0};
        // clang-format on
        vec1_size = gko::dim<2>(2, 2);
        nelems = 4;
    }
    vec1 = gko::matrix::Dense<value_type>::distributed_create(
        this->mpi_exec, vec1_size,
        gko::Array<value_type>::view(this->sub_exec, nelems, vec1_data), 2);
    vec1->compute_norm2(res1.get());

    EXPECT_EQ(res1->at(0, 0), T_nc(std::sqrt(68.25)));
    EXPECT_EQ(res1->at(0, 1), T_nc(std::sqrt(21.25)));
    delete vec1_data;
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
