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


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class MultiRankDistribute : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    MultiRankDistribute() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
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

    static void assert_equal_arrays(const gko::Array<value_type> &m,
                                    const gko::Array<value_type> &lm)
    {
        ASSERT_EQ(m.get_num_elems(), lm.get_num_elems());

        for (auto i = 0; i < m.get_num_elems(); ++i) {
            EXPECT_EQ(m.get_const_data()[i], lm.get_const_data()[i]);
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<const gko::Executor> sub_exec;
    int rank;
};

TYPED_TEST_CASE(MultiRankDistribute, gko::test::ValueTypes);


TYPED_TEST(MultiRankDistribute, CanSimpleDistributeArray)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> m{this->mpi_exec};
    gko::Array<value_type> orig_array{sub_exec};
    gko::Array<value_type> lm{sub_exec};
    gko::IndexSet<gko::int32> index_set{40};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        data = new value_type[28]{
            1.0,  2.0,  -1.0, 2.0, 0.0,  4.0,  -2.0,  // 6
            9.0,  -3.0, 7.0,  1.0, 0.2,  -3.5, 7.5,   // 13
            -1.1, 6.2,  1.0,  4.5, -1.5, 3.0,  9.0,   // 20
            4.0,  1.0,  3.5,  5.0, 6.0,  -1.0, 4.0    // 27
        };
        comp_data = new value_type[7]{
            1.0, 2.0, -1.0, 2.0, 0.0, 4.0, -2.0  // 6
        };
        orig_array = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 28, data));
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 7, comp_data));
        index_set.add_subset(0, 7);
    } else if (this->rank == 1) {
        comp_data = new value_type[7]{
            9.0, -3.0, 7.0, 1.0, 0.2, -3.5, 7.5  // 13
        };
        index_set.add_subset(7, 14);
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 7, comp_data));
    } else if (this->rank == 2) {
        comp_data = new value_type[7]{
            -1.1, 6.2, 1.0, 4.5, -1.5, 3.0, 9.0  // 20
        };
        index_set.add_subset(14, 21);
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 7, comp_data));
    } else if (this->rank == 3) {
        comp_data = new value_type[7]{
            4.0, 1.0, 3.5, 5.0, 6.0, -1.0, 4.0  // 27
        };
        index_set.add_subset(21, 28);
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 7, comp_data));
    }
    m = orig_array.distribute(this->mpi_exec, index_set);
    ASSERT_EQ(m.get_executor(), this->mpi_exec);
    this->assert_equal_arrays(m, lm);
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


TYPED_TEST(MultiRankDistribute, CanDistributeArrayNonContiguous)
{
    using value_type = typename TestFixture::value_type;
    value_type *data;
    value_type *comp_data;
    auto sub_exec = this->mpi_exec->get_sub_executor();
    gko::Array<value_type> m{this->mpi_exec};
    gko::Array<value_type> orig_array{sub_exec};
    gko::Array<value_type> lm{sub_exec};
    gko::IndexSet<gko::int32> index_set{40};
    this->mpi_exec->set_root_rank(0);
    if (this->rank == 0) {
        data = new value_type[28]{
            1.0,  2.0,  -1.0, 2.0, 0.0,  4.0,  -2.0,  // 6
            9.0,  -3.0, 7.0,  1.0, 0.2,  -3.5, 7.5,   // 13
            -1.1, 6.2,  1.0,  4.5, -1.5, 3.0,  9.0,   // 20
            4.0,  1.0,  3.5,  5.0, 6.0,  -1.0, 4.0    // 27
        };
        comp_data = new value_type[8]{2.0, 0.0, 9.0, -3.0, 7.0, 1.0, 0.2, -3.5};
        orig_array = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 28, data));
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 8, comp_data));
        index_set.add_subset(3, 5);
        index_set.add_subset(7, 13);
    } else if (this->rank == 1) {
        comp_data = new value_type[5]{2.0, -1.0, 7.5, -1.1, 6.2};
        index_set.add_subset(1, 3);
        index_set.add_subset(13, 16);
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 5, comp_data));
    } else if (this->rank == 2) {
        comp_data = new value_type[6]{1.0, 4.0, -2.0, 4.0, 1.0, 3.5};
        index_set.add_subset(5, 7);
        index_set.add_subset(0, 1);
        index_set.add_subset(21, 24);
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 6, comp_data));
    } else if (this->rank == 3) {
        comp_data =
            new value_type[9]{1.0, 4.5, -1.5, 3.0, 9.0, 5.0, 6.0, -1.0, 4.0};
        index_set.add_subset(16, 21);
        index_set.add_subset(24, 28);
        lm = gko::Array<value_type>(
            sub_exec, gko::Array<value_type>::view(sub_exec, 9, comp_data));
    }
    m = orig_array.distribute(this->mpi_exec, index_set);
    ASSERT_EQ(m.get_executor(), this->mpi_exec);
    this->assert_equal_arrays(m, lm);
    if (this->rank == 0) {
        delete data;
    }
    delete comp_data;
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
