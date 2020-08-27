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
#include <ginkgo/core/solver/ir.hpp>
#include <ginkgo/core/stop/combined.hpp>
#include <ginkgo/core/stop/iteration.hpp>
#include <ginkgo/core/stop/residual_norm.hpp>


#include "core/test/utils.hpp"


namespace {


template <typename T>
class DistributedIr : public ::testing::Test {
protected:
    using value_type = T;
    using Mtx = gko::matrix::Dense<value_type>;
    using Solver = gko::solver::Ir<value_type>;

    DistributedIr() : mpi_exec(nullptr) {}

    void SetUp()
    {
        char **argv;
        int argc = 0;
        exec = gko::ReferenceExecutor::create();
        mpi_exec = gko::MpiExecutor::create(gko::ReferenceExecutor::create());
        sub_exec = mpi_exec->get_sub_executor();
        rank = mpi_exec->get_my_rank();
        ASSERT_GT(mpi_exec->get_num_ranks(), 1);
        mtx = gko::initialize<Mtx>(
            {{2, -1.0, 0.0}, {-1.0, 2, -1.0}, {0.0, -1.0, 2}}, sub_exec);
        ir_factory =
            Solver::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(30u).on(
                        mpi_exec),
                    gko::stop::ResidualNormReduction<value_type>::build()
                        .with_reduction_factor(gko::remove_complex<T>{1e-6})
                        .on(mpi_exec))
                .on(mpi_exec);
        solver = ir_factory->generate(mtx);
    }

    void TearDown()
    {
        if (mpi_exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(mpi_exec->synchronize());
        }
    }

    std::shared_ptr<gko::MpiExecutor> mpi_exec;
    std::shared_ptr<const gko::Executor> exec;
    std::shared_ptr<const gko::Executor> sub_exec;
    std::shared_ptr<Mtx> mtx;
    std::unique_ptr<typename Solver::Factory> ir_factory;
    std::unique_ptr<gko::LinOp> solver;
    int rank;

    static void assert_same_matrices(const Mtx *m1, const Mtx *m2)
    {
        ASSERT_EQ(m1->get_size()[0], m2->get_size()[0]);
        ASSERT_EQ(m1->get_size()[1], m2->get_size()[1]);
        for (gko::size_type i = 0; i < m1->get_size()[0]; ++i) {
            for (gko::size_type j = 0; j < m2->get_size()[1]; ++j) {
                EXPECT_EQ(m1->at(i, j), m2->at(i, j));
            }
        }
    }
};

TYPED_TEST_CASE(DistributedIr, gko::test::ValueTypes);


TYPED_TEST(DistributedIr, DistributedIrFactoryKnowsItsExecutor)
{
    ASSERT_EQ(this->ir_factory->get_executor(), this->mpi_exec);
}


TYPED_TEST(DistributedIr, DistributedIrFactoryCreatesCorrectSolver)
{
    using Solver = typename TestFixture::Solver;

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(3, 3));
    auto ir_solver = static_cast<Solver *>(this->solver.get());
    ASSERT_NE(ir_solver->get_system_matrix(), nullptr);
    ASSERT_EQ(ir_solver->get_system_matrix(), this->mtx);
}


TYPED_TEST(DistributedIr, CanBeCopied)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->ir_factory->generate(Mtx::create(this->exec));

    copy->copy_from(this->solver.get());

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(DistributedIr, CanBeMoved)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto copy = this->ir_factory->generate(Mtx::create(this->exec));

    copy->copy_from(std::move(this->solver));

    ASSERT_EQ(copy->get_size(), gko::dim<2>(3, 3));
    auto copy_mtx = static_cast<Solver *>(copy.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(copy_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(DistributedIr, CanBeCloned)
{
    using Mtx = typename TestFixture::Mtx;
    using Solver = typename TestFixture::Solver;
    auto clone = this->solver->clone();

    ASSERT_EQ(clone->get_size(), gko::dim<2>(3, 3));
    auto clone_mtx = static_cast<Solver *>(clone.get())->get_system_matrix();
    this->assert_same_matrices(static_cast<const Mtx *>(clone_mtx.get()),
                               this->mtx.get());
}


TYPED_TEST(DistributedIr, CanBeCleared)
{
    using Solver = typename TestFixture::Solver;
    this->solver->clear();

    ASSERT_EQ(this->solver->get_size(), gko::dim<2>(0, 0));
    auto solver_mtx =
        static_cast<Solver *>(this->solver.get())->get_system_matrix();
    ASSERT_EQ(solver_mtx, nullptr);
}


TYPED_TEST(DistributedIr, ApplyUsesInitialGuessReturnsTrue)
{
    ASSERT_TRUE(this->solver->apply_uses_initial_guess());
}


TYPED_TEST(DistributedIr, CanSetCriteriaAgain)
{
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<gko::stop::CriterionFactory> init_crit =
        gko::stop::Iteration::build().with_max_iters(3u).on(this->exec);
    auto ir_factory = Solver::build().with_criteria(init_crit).on(this->exec);

    ASSERT_EQ((ir_factory->get_parameters().criteria).back(), init_crit);

    auto solver = ir_factory->generate(this->mtx);
    std::shared_ptr<gko::stop::CriterionFactory> new_crit =
        gko::stop::Iteration::build().with_max_iters(5u).on(this->exec);

    solver->set_stop_criterion_factory(new_crit);
    auto new_crit_fac = solver->get_stop_criterion_factory();
    auto niter =
        static_cast<const gko::stop::Iteration::Factory *>(new_crit_fac.get())
            ->get_parameters()
            .max_iters;

    ASSERT_EQ(niter, 5);
}

TYPED_TEST(DistributedIr, CanSolveIndependentLocalSystems)
{
    using Mtx = typename TestFixture::Mtx;
    using value_type = typename TestFixture::value_type;
    using Solver = typename TestFixture::Solver;
    std::shared_ptr<Solver> ir_precond =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->sub_exec)
            ->generate(this->mtx);
    auto b = gko::initialize<Mtx>({-1.0, 3.0, 1.0}, this->sub_exec);
    auto x = gko::initialize<Mtx>({0.0, 0.0, 0.0}, this->sub_exec);

    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->sub_exec);
    auto solver = ir_factory->generate(this->mtx);
    solver->apply(b.get(), x.get());

    GKO_ASSERT_MTX_NEAR(x, l({1.0, 3.0, 2.0}), r<value_type>::value);
}


TYPED_TEST(DistributedIr, CanSolveDistributedSystems)
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

    auto ir_factory =
        Solver::build()
            .with_criteria(gko::stop::Iteration::build().with_max_iters(3u).on(
                this->sub_exec))
            .on(this->mpi_exec);
    auto solver = ir_factory->generate(dist_mtx);
    solver->apply(b.get(), x.get());

    if (this->rank == 0) {
        GKO_ASSERT_MTX_NEAR(x, l({1.0, 2.0}), r<value_type>::value);
    } else {
        GKO_ASSERT_MTX_NEAR(x, l({3.0}), r<value_type>::value);
    }
}


}  // namespace

// Calls a custom gtest main with MPI listeners. See gtest-mpi-listeners.hpp for
// more details.
GKO_DECLARE_GTEST_MPI_MAIN;