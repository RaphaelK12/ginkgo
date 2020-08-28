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
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/fcg.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class DistributedFcg : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Fcg<value_type>;

    DistributedFcg() : mpi_exec(nullptr) {}

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
        mtx = gko::initialize<Mtx>(
            {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, sub_exec);
        fcg_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(30u).on(
                        mpi_exec),
                    gko::stop::ResidualNormReduction<value_type>::build()
                        .with_reduction_factor(gko::remove_complex<T>{1e-6})
                        .on(mpi_exec))
                .on(mpi_exec);
        solver = fcg_factory->generate(mtx);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<gko::MpiExecutor> mpi_exec2;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<gko::Executor> host;
    std::shared_ptr<const gko::Executor> sub_exec;
    std::shared_ptr<const gko::Executor> sub_exec2;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> fcg_factory;
    std::unique_ptr<gko::LinOp> solver;
    int rank;
};

TYPED_TEST_CASE(DistributedFcg, gko::test::ValueTypes);


TYPED_TEST(DistributedFcg, CanSolveIndependentLocalSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> fcg_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->sub_exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->sub_exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->sub_exec);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->sub_exec);
    auto solver = fcg_factory->generate(this->mtx);
    solver->set_preconditioner(fcg_precond);
    auto precond = solver->get_preconditioner();
    solver->apply(b.get(), x.get());

    auto x_h = Mtx::create(this->mpi_exec2);
    x_h->copy_from(x.get());

    GKO_ASSERT_MTX_NEAR(x_h, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(DistributedFcg, CanSolveDistributedSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using size_type = gko::size_type;
    using Solver = typename TestFixture::Solver;
    gko::IndexSet<size_type> row_dist{4};
    if (this->rank == 0) {
        row_dist.add_index(0);
        row_dist.add_index(2);
    } else {
        row_dist.add_index(1);
    }
    std::shared_ptr<Mtx> dist_mtx = gko::initialize_and_distribute<Mtx>(
        row_dist, {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}},
        this->mpi_exec);
    auto b = gko::initialize_and_distribute<Mtx>(1, row_dist, {-1.0, 3.0, 1.0},
                                                 this->mpi_exec);
    auto x = gko::initialize_and_distribute<Mtx>(1, row_dist, {0.0, 0.0, 0.0},
                                                 this->mpi_exec);

    auto fcg_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->mpi_exec);
    auto solver = fcg_factory->generate(dist_mtx);
    solver->apply(b.get(), x.get());

    auto x_h = Mtx::create(this->mpi_exec2);
    x_h->copy_from(x.get());
    if (this->rank == 0) {
        GKO_ASSERT_MTX_NEAR(x_h, l({1.0, 2.0}), r<value_type>::value);
    } else {
        GKO_ASSERT_MTX_NEAR(x_h, l({3.0}), r<value_type>::value);
    }
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;
