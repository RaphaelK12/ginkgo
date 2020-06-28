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

#ifndef GKO_CORE_BASE_EXECUTOR_HPP_
#define GKO_CORE_BASE_EXECUTOR_HPP_


#include <memory>
#include <mutex>
#include <sstream>
#include <tuple>
#include <type_traits>


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/machine_config.hpp>
#include <ginkgo/core/base/memory_space.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


struct cublasContext;

struct cusparseContext;

struct hipblasContext;

struct hipsparseContext;

struct machineInfoContext;


#ifndef MPI_VERSION

using MPI_Comm = int;
using MPI_Status = int;
using MPI_Request = int;
using MPI_Datatype = int;
using MPI_Op = int;

#ifndef MPI_COMM_WORLD
#define MPI_COMM_WORLD 0
#endif
#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF 0
#endif
#ifndef MPI_REQUEST_NULL
#define MPI_REQUEST_NULL 0
#endif
#ifndef MPI_MIN
#define MPI_MIN 0
#endif
#ifndef MPI_MAX
#define MPI_MAX 0
#endif
#ifndef MPI_SUM
#define MPI_SUM 0
#endif
#endif


namespace gko {


#define GKO_FORWARD_DECLARE(_type, ...) class _type

GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_FORWARD_DECLARE);

#undef GKO_FORWARD_DECLARE


class ReferenceExecutor;


namespace detail {


template <typename>
class ExecutorBase;


}  // namespace detail


/**
 * Operations can be used to define functionalities whose implementations differ
 * among devices.
 *
 * This is done by extending the Operation class and implementing the overloads
 * of the Operation::run() method for all Executor types. When invoking the
 * Executor::run() method with the Operation as input, the library will select
 * the Operation::run() overload corresponding to the dynamic type of the
 * Executor instance.
 *
 * Consider an overload of `operator<<` for Executors, which prints some basic
 * device information (e.g. device type and id) of the Executor to a C++ stream:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec);
 * ```
 *
 * One possible implementation would be to use RTTI to find the dynamic type of
 * the Executor, However, using the Operation feature of Ginkgo, there is a
 * more elegant approach which utilizes polymorphism. The first step is to
 * define an Operation that will print the desired information for each Executor
 * type.
 *
 * ```
 * class DeviceInfoPrinter : public gko::Operation {
 * public:
 *     explicit DeviceInfoPrinter(std::ostream &os) : os_(os) {}
 *
 *     void run(const gko::OmpExecutor *) const override { os_ << "OMP"; }
 *
 *     void run(const gko::MpiExecutor *) const override { os_ << "MPI"; }
 *
 *     void run(const gko::CudaExecutor *exec) const override
 *     { os_ << "CUDA(" << exec->get_device_id() << ")"; }
 *
 *     void run(const gko::HipExecutor *exec) const override
 *     { os_ << "HIP(" << exec->get_device_id() << ")"; }
 *
 *     // This is optional, if not overloaded, defaults to OmpExecutor overload
 *     void run(const gko::ReferenceExecutor *) const override
 *     { os_ << "Reference CPU"; }
 *
 * private:
 *     std::ostream &os_;
 * };
 * ```
 *
 * Using DeviceInfoPrinter, the implementation of `operator<<` is as simple as
 * calling the run() method of the executor.
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     DeviceInfoPrinter printer(os);
 *     exec.run(printer);
 *     return os;
 * }
 * ```
 *
 * Now it is possible to write the following code:
 *
 * ```
 * auto omp = gko::OmpExecutor::create();
 * std::cout << *omp << std::endl
 *           << *gko::CudaExecutor::create(0, omp) << std::endl
 *           << *gko::HipExecutor::create(0, omp) << std::endl
 *           << *gko::ReferenceExecutor::create() << std::endl;
 * ```
 *
 * which produces the expected output:
 *
 * ```
 * OMP
 * CUDA(0)
 * HIP(0)
 * Reference CPU
 * ```
 *
 * One might feel that this code is too complicated for such a simple task.
 * Luckily, there is an overload of the Executor::run() method, which is
 * designed to facilitate writing simple operations like this one. The method
 * takes three closures as input: one which is run for OMP, one for
 * CUDA executors, and the last one for HIP executors. Using this method, there
 * is no need to implement an Operation subclass:
 *
 * ```
 * std::ostream& operator<<(std::ostream &os, const gko::Executor &exec)
 * {
 *     exec.run(
 *         [&]() { os << "OMP"; },  // OMP closure
 *         [&]() { os << "MPI"; },  // MPI closure
 *         [&]() { os << "CUDA("    // CUDA closure
 *                    << static_cast<gko::CudaExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; },
 *         [&]() { os << "HIP("    // HIP closure
 *                    << static_cast<gko::HipExecutor&>(exec)
 *                         .get_device_id()
 *                    << ")"; });
 *     return os;
 * }
 * ```
 *
 * Using this approach, however, it is impossible to distinguish between
 * a OmpExecutor and ReferenceExecutor, as both of them call the OMP closure.
 *
 * @ingroup Executor
 */
class Operation {
public:
#define GKO_DECLARE_RUN_OVERLOAD(_type, ...) \
    virtual void run(std::shared_ptr<const _type>) const

    GKO_ENABLE_FOR_ALL_EXECUTORS(GKO_DECLARE_RUN_OVERLOAD);

#undef GKO_DECLARE_RUN_OVERLOAD

    // ReferenceExecutor overload can be defaulted to OmpExecutor's
    virtual void run(std::shared_ptr<const ReferenceExecutor> executor) const;

    /**
     * Returns the operation's name.
     *
     * @return the operation's name
     */
    virtual const char *get_name() const noexcept;
};

#define GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(_type, _namespace, _kernel)    \
public:                                                                      \
    void run(std::shared_ptr<const ::gko::_type> exec) const override        \
    {                                                                        \
        this->call(counts{}, exec);                                          \
    }                                                                        \
                                                                             \
private:                                                                     \
    template <int... Ns>                                                     \
    void call(::gko::syn::value_list<int, Ns...>,                            \
              std::shared_ptr<const ::gko::_type> &exec) const               \
    {                                                                        \
        ::gko::kernels::_namespace::_kernel(                                 \
            exec, std::forward<Args>(std::get<Ns>(data))...);                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_DETAIL_DEFINE_RUN_OVERLOAD(_type, _namespace, _kernel, ...)      \
public:                                                                      \
    void run(std::shared_ptr<const ::gko::_type> exec) const override        \
    {                                                                        \
        this->call(counts{}, exec);                                          \
    }                                                                        \
                                                                             \
private:                                                                     \
    template <int... Ns>                                                     \
    void call(::gko::syn::value_list<int, Ns...>,                            \
              std::shared_ptr<const ::gko::_type> &exec) const               \
    {                                                                        \
        ::gko::kernels::_namespace::_kernel(                                 \
            exec, std::forward<Args>(std::get<Ns>(data))...);                \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Binds a set of device-specific kernels to an Operation.
 *
 * It also defines a helper function which creates the associated operation.
 * Any input arguments passed to the helper function are forwarded to the
 * kernel when the operation is executed.
 *
 * The kernels used to bind the operation are searched in `kernels::DEV_TYPE`
 * namespace, where `DEV_TYPE` is replaced by `omp`, `mpi`, `cuda`, `hip` and
 * `reference`.
 *
 * @param _name  operation name
 * @param _kernel  kernel which will be bound to the operation
 *
 * Example
 * -------
 *
 * ```c++
 * // define the omp, mpi, cuda, hip and reference kernels which will be bound
 * to the
 * // operation
 * namespace kernels {
 * namespace omp {
 * void my_kernel(int x) {
 *      // omp code
 * }
 * }
 * namespace mpi {
 * void my_kernel(int x) {
 *      // mpi code
 * }
 * }
 * namespace cuda {
 * void my_kernel(int x) {
 *      // cuda code
 * }
 * }
 * namespace hip {
 * void my_kernel(int x) {
 *      // hip code
 * }
 * }
 * namespace reference {
 * void my_kernel(int x) {
 *     // reference code
 * }
 * }
 *
 * // Bind the kernels to the operation
 * GKO_REGISTER_OPERATION(my_op, my_kernel);
 *
 * int main() {
 *     // create executors
 *     auto omp = OmpExecutor::create();
 *     auto mpi = MpiExecutor::create();
 *     auto cuda = CudaExecutor::create(omp, 0);
 *     auto hip = HipExecutor::create(omp, 0);
 *     auto ref = ReferenceExecutor::create();
 *
 *     // create the operation
 *     auto op = make_my_op(5); // x = 5
 *
 *     omp->run(op);  // run omp kernel
 *     mpi->run(op);  // run mpi kernel
 *     cuda->run(op);  // run cuda kernel
 *     hip->run(op);  // run hip kernel
 *     ref->run(op);  // run reference kernel
 * }
 * ```
 *
 * @ingroup Executor
 */
#define GKO_REGISTER_OPERATION(_name, _kernel)                               \
    template <typename... Args>                                              \
    class _name##_operation : public Operation {                             \
        using counts =                                                       \
            ::gko::syn::as_list<::gko::syn::range<0, sizeof...(Args)>>;      \
                                                                             \
    public:                                                                  \
        explicit _name##_operation(Args &&... args)                          \
            : data(std::forward<Args>(args)...)                              \
        {}                                                                   \
                                                                             \
        const char *get_name() const noexcept override                       \
        {                                                                    \
            static auto name = [this] {                                      \
                std::ostringstream oss;                                      \
                oss << #_kernel << '#' << sizeof...(Args);                   \
                return oss.str();                                            \
            }();                                                             \
            return name.c_str();                                             \
        }                                                                    \
                                                                             \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(OmpExecutor, omp, _kernel);    \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(CudaExecutor, cuda, _kernel);  \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(HipExecutor, hip, _kernel);    \
        GKO_KERNEL_DETAIL_DEFINE_RUN_OVERLOAD(ReferenceExecutor, reference,  \
                                              _kernel);                      \
                                                                             \
    private:                                                                 \
        mutable std::tuple<Args &&...> data;                                 \
    };                                                                       \
                                                                             \
    template <typename... Args>                                              \
    static _name##_operation<Args...> make_##_name(Args &&... args)          \
    {                                                                        \
        return _name##_operation<Args...>(std::forward<Args>(args)...);      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * The first step in using the Ginkgo library consists of creating an
 * executor. Executors are used to specify the location for the data of linear
 * algebra objects, and to determine where the operations will be executed.
 * Ginkgo currently supports three different executor types:
 *
 * +    OmpExecutor specifies that the data should be stored and the associated
 *      operations executed on an OpenMP-supporting device (e.g. host CPU);
 * +    MpiExecutor specifies that the data should be stored and the associated
 *      operations executed on an MPI-supporting device (e.g. host CPU);
 * +    CudaExecutor specifies that the data should be stored and the
 *      operations executed on the NVIDIA GPU accelerator;
 * +    HipExecutor specifies that the data should be stored and the
 *      operations executed on either an NVIDIA or AMD GPU accelerator;
 * +    ReferenceExecutor executes a non-optimized reference implementation,
 *      which can be used to debug the library.
 *
 * The following code snippet demonstrates the simplest possible use of the
 * Ginkgo library:
 *
 * ```cpp
 * auto omp = gko::create<gko::OmpExecutor>();
 * auto A = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", omp);
 * ```
 *
 * First, we create a OMP executor, which will be used in the next line to
 * specify where we want the data for the matrix A to be stored.
 * The second line will read a matrix from the matrix market file 'A.mtx',
 * and store the data on the CPU in CSR format (gko::matrix::Csr is a
 * Ginkgo matrix class which stores its data in CSR format).
 * At this point, matrix A is bound to the CPU, and any routines called on it
 * will be performed on the CPU. This approach is usually desired in sparse
 * linear algebra, as the cost of individual operations is several orders of
 * magnitude lower than the cost of copying the matrix to the GPU.
 *
 * If matrix A is going to be reused multiple times, it could be beneficial to
 * copy it over to the accelerator, and perform the operations there, as
 * demonstrated by the next code snippet:
 *
 * ```cpp
 * auto cuda = gko::create<gko::CudaExecutor>(0, omp);
 * auto dA = gko::copy_to<gko::matrix::Csr<float>>(A.get(), cuda);
 * ```
 *
 * The first line of the snippet creates a new CUDA executor. Since there may be
 * multiple NVIDIA GPUs present on the system, the first parameter instructs the
 * library to use the first device (i.e. the one with device ID zero, as in
 * cudaSetDevice() routine from the CUDA runtime API). In addition, since GPUs
 * are not stand-alone processors, it is required to pass a "master" OmpExecutor
 * which will be used to schedule the requested CUDA kernels on the accelerator.
 *
 * The second command creates a copy of the matrix A on the GPU. Notice the use
 * of the get() method. As Ginkgo aims to provide automatic memory
 * management of its objects, the result of calling gko::read_from_mtx()
 * is a smart pointer (std::unique_ptr) to the created object. On the other
 * hand, as the library will not hold a reference to A once the copy is
 * completed, the input parameter for gko::copy_to() is a plain pointer.
 * Thus, the get() method is used to convert from a std::unique_ptr to a
 * plain pointer, as expected by gko::copy_to().
 *
 * As a side note, the gko::copy_to routine is far more powerful than just
 * copying data between different devices. It can also be used to convert data
 * between different formats. For example, if the above code used
 * gko::matrix::Ell as the template parameter, dA would be stored on the GPU,
 * in ELLPACK format.
 *
 * Finally, if all the processing of the matrix is supposed to be done on the
 * GPU, and a CPU copy of the matrix is not required, we could have read the
 * matrix to the GPU directly:
 *
 * ```cpp
 * auto omp = gko::create<gko::OmpExecutor>();
 * auto cuda = gko::create<gko::CudaExecutor>(0, omp);
 * auto dA = gko::read_from_mtx<gko::matrix::Csr<float>>("A.mtx", cuda);
 * ```
 * Notice that even though reading the matrix directly from a file to the
 * accelerator is not supported, the library is designed to abstract away the
 * intermediate step of reading the matrix to the CPU memory. This is a general
 * design approach taken by the library: in case an operation is not supported
 * by the device, the data will be copied to the CPU, the operation performed
 * there, and finally the results copied back to the device.
 * This approach makes using the library more concise, as explicit copies are
 * not required by the user. Nevertheless, this feature should be taken into
 * account when considering performance implications of using such operations.
 *
 * @ingroup Executor
 */
class Executor : public log::EnableLogging<Executor> {
    template <typename T>
    friend class detail::ExecutorBase;

public:
    virtual ~Executor() = default;

    Executor() = default;
    Executor(Executor &) = delete;
    Executor(Executor &&) = default;
    Executor &operator=(Executor &) = delete;
    Executor &operator=(Executor &&) = default;

    /**
     * Runs the specified Operation using this Executor.
     *
     * @param op  the operation to run
     */
    virtual void run(const Operation &op) const = 0;

    /**
     * Runs one of the passed in functors, depending on the Executor type.
     *
     * @tparam ClosureOmp  type of op_omp
     * @tparam ClosureMpi  type of op_mpi
     * @tparam ClosureCuda  type of op_cuda
     * @tparam ClosureHip  type of op_hip
     *
     * @param op_omp  functor to run in case of a OmpExecutor or
     *                ReferenceExecutor
     * @param op_mpi  functor to run in case of a MpiExecutor
     * @param op_cuda  functor to run in case of a CudaExecutor
     * @param op_hip  functor to run in case of a HipExecutor
     */
    template <typename ClosureOmp, typename ClosureMpi, typename ClosureCuda,
              typename ClosureHip>
    void run(const ClosureOmp &op_omp, const ClosureMpi &op_mpi,
             const ClosureCuda &op_cuda, const ClosureHip &op_hip) const
    {
        LambdaOperation<ClosureOmp, ClosureMpi, ClosureCuda, ClosureHip> op(
            op_omp, op_mpi, op_cuda, op_hip);
        this->run(op);
    }

    /**
     * Copies data within this Executor.
     *
     * @tparam T  datatype to copy
     *
     * @param num_elems  number of elements of type T to copy
     * @param src_ptr  pointer to a block of memory containing the data to be
     *                 copied
     * @param dest_ptr  pointer to an allocated block of memory
     *                  where the data will be copied to
     */
    template <typename T>
    void copy(size_type num_elems, const T *src_ptr, T *dest_ptr) const
    {
        this->get_mem_space()->copy_from(this->get_mem_space().get(), num_elems,
                                         src_ptr, dest_ptr);
    }

    /**
     * Retrieves a single element at the given location from executor memory.
     *
     * @tparam T  datatype to copy
     *
     * @param ptr  the pointer to the element to be copied
     *
     * @return the value stored at ptr
     */
    template <typename T>
    T copy_val_to_host(const T *ptr) const
    {
        T out{};
        this->get_master()->get_mem_space()->copy_from(
            this->get_mem_space().get(), 1, ptr, &out);
        return out;
    }

    /**
     * Returns the master OmpExecutor of this Executor.
     * @return the master OmpExecutor of this Executor.
     */
    virtual std::shared_ptr<Executor> get_master() noexcept = 0;

    /**
     * @copydoc get_master
     */
    virtual std::shared_ptr<const Executor> get_master() const noexcept = 0;

    /**
     * Returns the sub-executor of this Executor.
     * @return the sub-executor of this Executor.
     */
    virtual std::shared_ptr<Executor> get_sub_executor() noexcept = 0;

    /**
     * @copydoc get_sub_executor
     */
    virtual std::shared_ptr<const Executor> get_sub_executor() const
        noexcept = 0;

    /**
     * Returns the associated memory space of this Executor.
     * @return the associated memory space of this Executor.
     */
    virtual std::shared_ptr<MemorySpace> get_mem_space() noexcept = 0;

    /**
     * @copydoc get_mem_space
     */
    virtual std::shared_ptr<const MemorySpace> get_mem_space() const
        noexcept = 0;

    /**
     * Synchronize the operations launched on the executor with its master.
     */
    virtual void synchronize() const = 0;

private:
    /**
     * The LambdaOperation class wraps four functor objects into an
     * Operation.
     *
     * The first object is called by the OmpExecutor, the second one by the
     * CudaExecutor and the last one by the HipExecutor. When run on the
     * ReferenceExecutor, the implementation will launch the CPU reference
     * version.
     *
     * @tparam ClosureOmp  the type of the first functor
     * @tparam ClosureMpi  the type of the second functor
     * @tparam ClosureCuda  the type of the second functor
     * @tparam ClosureHip  the type of the third functor
     */
    template <typename ClosureOmp, typename ClosureMpi, typename ClosureCuda,
              typename ClosureHip>
    class LambdaOperation : public Operation {
    public:
        /**
         * Creates an LambdaOperation object from two functors.
         *
         * @param op_omp  a functor object which will be called by OmpExecutor
         *                and ReferenceExecutor
         * @param op_mpi  a functor object which will be called by MpiExecutor
         * @param op_cuda  a functor object which will be called by CudaExecutor
         * @param op_hip  a functor object which will be called by HipExecutor
         */
        LambdaOperation(const ClosureOmp &op_omp, const ClosureMpi &op_mpi,
                        const ClosureCuda &op_cuda, const ClosureHip &op_hip)
            : op_omp_(op_omp),
              op_mpi_(op_mpi),
              op_cuda_(op_cuda),
              op_hip_(op_hip)
        {}

        void run(std::shared_ptr<const OmpExecutor>) const override
        {
            op_omp_();
        }

        void run(std::shared_ptr<const MpiExecutor>) const override
        {
            op_mpi_();
        }

        void run(std::shared_ptr<const CudaExecutor>) const override
        {
            op_cuda_();
        }

        void run(std::shared_ptr<const HipExecutor>) const override
        {
            op_hip_();
        }

    private:
        ClosureOmp op_omp_;
        ClosureMpi op_mpi_;
        ClosureCuda op_cuda_;
        ClosureHip op_hip_;
    };
};


namespace detail {


template <typename ConcreteExecutor>
class ExecutorBase : public Executor {
public:
    void run(const Operation &op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        op.run(self()->shared_from_this());
        this->template log<log::Logger::operation_completed>(this, &op);
    }

private:
    ConcreteExecutor *self() noexcept
    {
        return static_cast<ConcreteExecutor *>(this);
    }

    const ConcreteExecutor *self() const noexcept
    {
        return static_cast<const ConcreteExecutor *>(this);
    }
};


/**
 * Controls whether the DeviceReset function should be called thanks to a
 * boolean. Note that in any case, `DeviceReset` is called only after destroying
 * the last Ginkgo executor. Therefore, it is sufficient to set this flag to the
 * last living executor in Ginkgo. Setting this flag to an executor which is not
 * destroyed last has no effect.
 */
class EnableDeviceReset {
public:
    /**
     * Set the device reset capability.
     *
     * @param device_reset  whether to allow a device reset or not
     */
    void set_device_reset(bool device_reset) { device_reset_ = device_reset; }

    /**
     * Returns the current status of the device reset boolean for this executor.
     *
     * @return the current status of the device reset boolean for this executor.
     */
    bool get_device_reset() { return device_reset_; }

protected:
    /**
     * Instantiate an EnableDeviceReset class
     *
     * @param device_reset  the starting device_reset status. Defaults to false.
     */
    EnableDeviceReset(bool device_reset = false) : device_reset_{device_reset}
    {}

private:
    bool device_reset_{};
};


}  // namespace detail


/**
 * This is the Executor subclass which represents the OpenMP device
 * (typically CPU).
 *
 * @ingroup exec_omp
 * @ingroup Executor
 */
class OmpExecutor : public detail::ExecutorBase<OmpExecutor>,
                    public std::enable_shared_from_this<OmpExecutor>,
                    public machine_config::topology<OmpExecutor> {
    friend class detail::ExecutorBase<OmpExecutor>;

public:
    using omp_exec_info = machine_config::topology<OmpExecutor>;

    using DefaultMemorySpace = HostMemorySpace;

    /**
     * Creates a new OmpExecutor.
     */
    static std::shared_ptr<OmpExecutor> create()
    {
        return std::shared_ptr<OmpExecutor>(new OmpExecutor());
    }

    /**
     * Creates a new OmpExecutor with an existing memory space.
     *
     * @param memory_space  The memory space to be associated with the executor.
     */
    static std::shared_ptr<OmpExecutor> create(
        std::shared_ptr<MemorySpace> memory_space)
    {
        return std::shared_ptr<OmpExecutor>(new OmpExecutor(memory_space));
    }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    std::shared_ptr<Executor> get_sub_executor() noexcept override;

    std::shared_ptr<const Executor> get_sub_executor() const noexcept override;

    std::shared_ptr<MemorySpace> get_mem_space() noexcept override;

    std::shared_ptr<const MemorySpace> get_mem_space() const noexcept override;

    void synchronize() const override;

    /**
     * Get the Executor information for this executor
     *
     * @return the executor info (omp_exec_info*) for this executor
     */
    omp_exec_info *get_exec_info() const { return exec_info_.get(); }

protected:
    OmpExecutor() : exec_info_(omp_exec_info::create())
    {
        mem_space_instance_ = HostMemorySpace::create();
    }

    OmpExecutor(std::shared_ptr<MemorySpace> mem_space)
        : mem_space_instance_(mem_space)
    {
        if (!check_mem_space_validity(mem_space_instance_)) {
            GKO_MEMSPACE_MISMATCH(NOT_HOST);
        }
    }

    bool check_mem_space_validity(std::shared_ptr<MemorySpace> mem_space)
    {
        auto check_default_mem_space =
            dynamic_cast<DefaultMemorySpace *>(mem_space.get());
        if (check_default_mem_space == nullptr) {
            return false;
        } else {
            return true;
        }
    }

private:
    std::unique_ptr<omp_exec_info> exec_info_;
    std::shared_ptr<MemorySpace> mem_space_instance_;
};


namespace kernels {
namespace omp {
using DefaultExecutor = OmpExecutor;
}  // namespace omp
}  // namespace kernels


/**
 * This is the Executor subclass which represents the MPI device
 * (typically CPU).
 *
 * @ingroup exec_mpi
 * @ingroup Executor
 */
class MpiExecutor : public detail::ExecutorBase<MpiExecutor>,
                    public std::enable_shared_from_this<MpiExecutor>,
                    public machine_config::topology<MpiExecutor> {
    friend class detail::ExecutorBase<MpiExecutor>;

public:
    using mpi_exec_info = machine_config::topology<MpiExecutor>;
    template <typename T>
    using request_manager = std::unique_ptr<T, std::function<void(T *)>>;

    using DefaultMemorySpace = DistributedMemorySpace;

    /**
     * Creates a new MpiExecutor.
     */
    static std::shared_ptr<MpiExecutor> create(
        std::initializer_list<std::string> sub_exec_list, int num_args = 0,
        char **args = nullptr);

    static std::shared_ptr<MpiExecutor> create();

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    std::shared_ptr<Executor> get_sub_executor() noexcept override;

    std::shared_ptr<const Executor> get_sub_executor() const noexcept override;

    void run(const Operation &op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        op.run(std::static_pointer_cast<const MpiExecutor>(
            this->shared_from_this()));
        this->template log<log::Logger::operation_completed>(this, &op);
    }

    int get_num_ranks() const;

    int get_my_rank() const;

    MPI_Comm get_communicator() const { return mpi_comm_; }

    void set_root_rank(int rank) { root_rank_ = rank; }

    int get_root_rank() const { return root_rank_; }

    std::shared_ptr<MemorySpace> get_mem_space() noexcept override;

    std::shared_ptr<const MemorySpace> get_mem_space() const noexcept override;

    void synchronize() const override;

    void synchronize_communicator(MPI_Comm &comm) const;

    MPI_Comm create_communicator(MPI_Comm &comm, int color, int key);

    request_manager<MPI_Request> create_requests_array(int size);

    /**
     * Get the Executor information for this executor
     *
     * @return the executor info (mpi_exec_info*) for this executor
     */
    mpi_exec_info *get_exec_info() const { return exec_info_.get(); }

    std::vector<std::string> get_sub_executor_list() const
    {
        return sub_exec_list_;
    }

    // MPI_Send
    template <typename SendType>
    void send(const SendType *send_buffer, const int send_count,
              const int destination_rank, const int send_tag,
              bool non_blocking = false);

    // MPI_Recv
    template <typename RecvType>
    void recv(RecvType *recv_buffer, const int recv_count,
              const int source_rank, const int recv_tag,
              bool non_blocking = false);

    // MPI_Gather
    template <typename SendType, typename RecvType>
    void gather(const SendType *send_buffer, const int send_count,
                RecvType *recv_buffer, const int recv_count, int root_rank);

    // MPI_Gatherv
    template <typename SendType, typename RecvType>
    void gather(const SendType *send_buffer, const int send_count,
                RecvType *recv_buffer, const int *recv_counts,
                const int *displacements, int root_rank);

    // MPI_Scatter
    template <typename SendType, typename RecvType>
    void scatter(const SendType *send_buffer, const int send_count,
                 RecvType *recv_buffer, const int recv_count, int root_rank);

    // MPI_Scatterv
    template <typename SendType, typename RecvType>
    void scatter(const SendType *send_buffer, const int *send_counts,
                 const int *displacements, RecvType *recv_buffer,
                 const int recv_count, int root_rank);

    // MPI_Bcast
    template <typename BroadcastType>
    void broadcast(BroadcastType *buffer, int count, int root_rank);

protected:
    MpiExecutor() = delete;

    void mpi_init();

    void create_sub_executors(std::vector<std::string> &sub_exec_list,
                              std::shared_ptr<gko::Executor> &sub_executor);

    bool is_finalized() const;

    bool is_initialized() const;

    void destroy();

    MpiExecutor(std::initializer_list<std::string> sub_exec_list, int num_args,
                char **args)
        : num_ranks_(1),
          num_args_(num_args),
          args_(args),
          sub_exec_list_(sub_exec_list)
    {
        this->mpi_init();
        num_ranks_ = this->get_num_ranks();
        root_rank_ = 0;
        this->create_sub_executors(sub_exec_list_, sub_executor_);
    }

    bool check_mem_space_validity(std::shared_ptr<MemorySpace> mem_space)
    {
        auto check_default_mem_space =
            dynamic_cast<DefaultMemorySpace *>(mem_space.get());
        if (check_default_mem_space == nullptr) {
            return false;
        } else {
            return true;
        }
    }

private:
    int num_ranks_;
    int num_args_;
    int root_rank_;
    int required_thread_support_;
    int provided_thread_support_;
    char **args_;
    std::vector<std::string> sub_exec_list_;
    std::shared_ptr<Executor> sub_executor_;

    MPI_Comm mpi_comm_;
    template <typename T>
    using status_manager = std::unique_ptr<T, std::function<void(T *)>>;
    status_manager<MPI_Status> mpi_status_;

    std::unique_ptr<mpi_exec_info> exec_info_;
    std::shared_ptr<MemorySpace> mem_space_instance_;
};


namespace kernels {
namespace mpi {
using DefaultExecutor = MpiExecutor;
}  // namespace mpi
}  // namespace kernels


/**
 * This is a specialization of the OmpExecutor, which runs the reference
 * implementations of the kernels used for debugging purposes.
 *
 * @ingroup exec_ref
 * @ingroup Executor
 */
class ReferenceExecutor : public OmpExecutor {
public:
    using ref_exec_info = machine_config::topology<OmpExecutor>;

    /**
     * Creates a new ReferenceExecutor with an existing memory space.
     *
     */
    static std::shared_ptr<ReferenceExecutor> create()
    {
        return std::shared_ptr<ReferenceExecutor>(new ReferenceExecutor());
    }

    /**
     * Creates a new ReferenceExecutor with an existing memory space.
     *
     * @param memory_space  The memory space to be associated with the executor.
     */
    static std::shared_ptr<ReferenceExecutor> create(
        std::shared_ptr<MemorySpace> memory_space)
    {
        return std::shared_ptr<ReferenceExecutor>(
            new ReferenceExecutor(memory_space));
    }

    void run(const Operation &op) const override
    {
        this->template log<log::Logger::operation_launched>(this, &op);
        op.run(std::static_pointer_cast<const ReferenceExecutor>(
            this->shared_from_this()));
        this->template log<log::Logger::operation_completed>(this, &op);
    }

    /**
     * Get the Executor information for this executor
     *
     * @return the executor info (ref_exec_info*) for this executor
     */
    ref_exec_info *get_exec_info() const { return exec_info_.get(); }

    std::shared_ptr<MemorySpace> get_mem_space() noexcept override
    {
        return this->mem_space_instance_;
    }

    std::shared_ptr<const MemorySpace> get_mem_space() const noexcept override
    {
        return this->mem_space_instance_;
    }

protected:
    ReferenceExecutor() : exec_info_(ref_exec_info::create())
    {
        mem_space_instance_ = HostMemorySpace::create();
    }

    ReferenceExecutor(std::shared_ptr<MemorySpace> mem_space)
        : mem_space_instance_(mem_space)
    {
        if (!check_mem_space_validity(mem_space_instance_)) {
            GKO_MEMSPACE_MISMATCH(NOT_HOST);
        }
    }

    bool check_mem_space_validity(std::shared_ptr<MemorySpace> mem_space)
    {
        auto check_default_mem_space =
            dynamic_cast<DefaultMemorySpace *>(mem_space.get());
        if (check_default_mem_space == nullptr) {
            return false;
        } else {
            return true;
        }
    }


private:
    std::unique_ptr<ref_exec_info> exec_info_;
    std::shared_ptr<MemorySpace> mem_space_instance_;
};


namespace kernels {
namespace reference {
using DefaultExecutor = ReferenceExecutor;
}  // namespace reference
}  // namespace kernels


/**
 * This is the Executor subclass which represents the CUDA device.
 *
 * @ingroup exec_cuda
 * @ingroup Executor
 */
class CudaExecutor : public detail::ExecutorBase<CudaExecutor>,
                     public std::enable_shared_from_this<CudaExecutor>,
                     public detail::EnableDeviceReset,
                     public machine_config::topology<CudaExecutor> {
    friend class detail::ExecutorBase<CudaExecutor>;

public:
    using cuda_exec_info = machine_config::topology<CudaExecutor>;
    using DefaultMemorySpace = CudaMemorySpace;

    /**
     * Creates a new CudaExecutor.
     *
     * @param device_id  the CUDA device id of this device
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     */
    static std::shared_ptr<CudaExecutor> create(
        int device_id, std::shared_ptr<Executor> master,
        bool device_reset = false);

    /**
     * Creates a new CudaExecutor.
     *
     * @param device_id  the CUDA device id of this device
     * @param memory_space  the memory space associated to the executor.
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     */
    static std::shared_ptr<CudaExecutor> create(
        int device_id, std::shared_ptr<MemorySpace> memory_space,
        std::shared_ptr<Executor> master);

    ~CudaExecutor() { decrease_num_execs(this->device_id_); }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    std::shared_ptr<Executor> get_sub_executor() noexcept override;

    std::shared_ptr<const Executor> get_sub_executor() const noexcept override;

    std::shared_ptr<MemorySpace> get_mem_space() noexcept override;

    std::shared_ptr<const MemorySpace> get_mem_space() const noexcept override;

    void synchronize() const override;

    void run(const Operation &op) const override;

    /**
     * Get the CUDA device id of the device associated to this executor.
     */
    int get_device_id() const noexcept { return device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    /**
     * Get the number of warps per SM of this executor.
     */
    int get_num_warps_per_sm() const noexcept { return num_warps_per_sm_; }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept { return num_multiprocessor_; }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        return num_multiprocessor_ * num_warps_per_sm_;
    }

    /**
     * Get the warp size of this executor.
     */
    int get_warp_size() const noexcept { return warp_size_; }

    /**
     * Get the major verion of compute capability.
     */
    int get_major_version() const noexcept { return major_; }

    /**
     * Get the minor verion of compute capability.
     */
    int get_minor_version() const noexcept { return minor_; }

    /**
     * Get the cublas handle for this executor
     *
     * @return  the cublas handle (cublasContext*) for this executor
     */
    cublasContext *get_cublas_handle() const { return cublas_handle_.get(); }

    /**
     * Get the cusparse handle for this executor
     *
     * @return the cusparse handle (cusparseContext*) for this executor
     */
    cusparseContext *get_cusparse_handle() const
    {
        return cusparse_handle_.get();
    }

    /**
     * Get the Executor information for this executor
     *
     * @return the executor info (cuda_exec_info*) for this executor
     */
    cuda_exec_info *get_exec_info() const { return exec_info_.get(); }

protected:
    void set_gpu_property();

    void init_handles();

    CudaExecutor(int device_id, std::shared_ptr<Executor> master,
                 bool device_reset = false)
        : EnableDeviceReset{device_reset},
          device_id_(device_id),
          master_(master),
          num_warps_per_sm_(0),
          num_multiprocessor_(0),
          major_(0),
          minor_(0),
          warp_size_(0)
    {
        assert(device_id < max_devices && device_id >= 0);
        this->set_gpu_property();
        this->init_handles();
        increase_num_execs(device_id);
        exec_info_ = cuda_exec_info::create();
        mem_space_instance_ = CudaMemorySpace::create(device_id);
    }

    CudaExecutor(int device_id, std::shared_ptr<MemorySpace> mem_space,
                 std::shared_ptr<Executor> master)
        : device_id_(device_id),
          master_(master),
          num_warps_per_sm_(0),
          num_multiprocessor_(0),
          major_(0),
          minor_(0),
          mem_space_instance_(mem_space)
    {
        assert(device_id < max_devices);
        this->set_gpu_property();
        this->init_handles();
        increase_num_execs(device_id);
        if (!check_mem_space_validity(mem_space_instance_)) {
            GKO_MEMSPACE_MISMATCH(NOT_CUDA);
        }
    }

    bool check_mem_space_validity(std::shared_ptr<MemorySpace> mem_space)
    {
        auto check_cuda_mem_space =
            dynamic_cast<CudaMemorySpace *>(mem_space.get());
        auto check_cuda_uvm_mem_space =
            dynamic_cast<CudaUVMSpace *>(mem_space.get());
        if (check_cuda_mem_space == nullptr &&
            check_cuda_uvm_mem_space == nullptr) {
            return false;
        }
        return true;
    }

    static void increase_num_execs(unsigned device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]++;
    }

    static void decrease_num_execs(unsigned device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]--;
    }

    static unsigned get_num_execs(unsigned device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        return num_execs[device_id];
    }

private:
    int device_id_;
    std::shared_ptr<Executor> master_;
    int num_warps_per_sm_;
    int num_multiprocessor_;
    int major_;
    int minor_;
    int warp_size_;
    std::unique_ptr<cuda_exec_info> exec_info_;
    std::shared_ptr<MemorySpace> mem_space_instance_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<cublasContext> cublas_handle_;
    handle_manager<cusparseContext> cusparse_handle_;

    static constexpr int max_devices = 64;
    static unsigned num_execs[max_devices];
    static std::mutex mutex[max_devices];
};


namespace kernels {
namespace cuda {
using DefaultExecutor = CudaExecutor;
}  // namespace cuda
}  // namespace kernels


/**
 * This is the Executor subclass which represents the HIP enhanced device.
 *
 * @ingroup exec_hip
 * @ingroup Executor
 */
class HipExecutor : public detail::ExecutorBase<HipExecutor>,
                    public std::enable_shared_from_this<HipExecutor>,
                    public detail::EnableDeviceReset,
                    public machine_config::topology<HipExecutor> {
    friend class detail::ExecutorBase<HipExecutor>;

public:
    using hip_exec_info = machine_config::topology<HipExecutor>;
    using DefaultMemorySpace = HipMemorySpace;

    /**
     * Creates a new HipExecutor.
     *
     * @param device_id  the HIP device id of this device
     * @param master  an executor on the host that is used to invoke the device
     *                kernels
     */
    static std::shared_ptr<HipExecutor> create(int device_id,
                                               std::shared_ptr<Executor> master,
                                               bool device_reset = false);

    /**
     * Creates a new HipExecutor.
     *
     * @param device_id  the HIP device id of this device
     * @param memory_space  the memory space associated to the executor.
     * @param master  an executor on the host that is used to invoke the device
     * kernels
     */
    static std::shared_ptr<HipExecutor> create(
        int device_id, std::shared_ptr<MemorySpace> memory_space,
        std::shared_ptr<Executor> master);

    ~HipExecutor() { decrease_num_execs(this->device_id_); }

    std::shared_ptr<Executor> get_master() noexcept override;

    std::shared_ptr<const Executor> get_master() const noexcept override;

    std::shared_ptr<Executor> get_sub_executor() noexcept override;

    std::shared_ptr<const Executor> get_sub_executor() const noexcept override;

    std::shared_ptr<MemorySpace> get_mem_space() noexcept override;

    std::shared_ptr<const MemorySpace> get_mem_space() const noexcept override;

    void synchronize() const override;

    void run(const Operation &op) const override;

    /**
     * Get the HIP device id of the device associated to this executor.
     */
    int get_device_id() const noexcept { return device_id_; }

    /**
     * Get the number of devices present on the system.
     */
    static int get_num_devices();

    /**
     * Get the number of warps per SM of this executor.
     */
    int get_num_warps_per_sm() const noexcept { return num_warps_per_sm_; }

    /**
     * Get the number of multiprocessor of this executor.
     */
    int get_num_multiprocessor() const noexcept { return num_multiprocessor_; }

    /**
     * Get the major verion of compute capability.
     */
    int get_major_version() const noexcept { return major_; }

    /**
     * Get the minor verion of compute capability.
     */
    int get_minor_version() const noexcept { return minor_; }

    /**
     * Get the number of warps of this executor.
     */
    int get_num_warps() const noexcept
    {
        return num_multiprocessor_ * num_warps_per_sm_;
    }

    /**
     * Get the warp size of this executor.
     */
    int get_warp_size() const noexcept { return warp_size_; }

    /**
     * Get the hipblas handle for this executor
     *
     * @return  the hipblas handle (hipblasContext*) for this executor
     */
    hipblasContext *get_hipblas_handle() const { return hipblas_handle_.get(); }

    /**
     * Get the hipsparse handle for this executor
     *
     * @return the hipsparse handle (hipsparseContext*) for this executor
     */
    hipsparseContext *get_hipsparse_handle() const
    {
        return hipsparse_handle_.get();
    }

    /**
     * Get the Executor information for this executor
     *
     * @return the executor info (cuda_exec_info*) for this executor
     */
    hip_exec_info *get_exec_info() const { return exec_info_.get(); }

protected:
    void set_gpu_property();

    void init_handles();

    HipExecutor(int device_id, std::shared_ptr<Executor> master,
                bool device_reset = false)
        : EnableDeviceReset{device_reset},
          device_id_(device_id),
          master_(master),
          num_multiprocessor_(0),
          num_warps_per_sm_(0),
          major_(0),
          minor_(0),
          warp_size_(0)
    {
        assert(device_id < max_devices);
        this->set_gpu_property();
        this->init_handles();
        increase_num_execs(device_id);
        exec_info_ = hip_exec_info::create();
        mem_space_instance_ = HipMemorySpace::create(device_id);
    }

    HipExecutor(int device_id, std::shared_ptr<MemorySpace> mem_space,
                std::shared_ptr<Executor> master)
        : device_id_(device_id),
          master_(master),
          num_multiprocessor_(0),
          mem_space_instance_(mem_space)
    {
        assert(device_id < max_devices);
        this->set_gpu_property();
        this->init_handles();
        increase_num_execs(device_id);
        if (!check_mem_space_validity(mem_space_instance_)) {
            GKO_MEMSPACE_MISMATCH(NOT_HIP);
        }
    }

    bool check_mem_space_validity(std::shared_ptr<MemorySpace> mem_space)
    {
        auto check_hip_mem_space =
            dynamic_cast<HipMemorySpace *>(mem_space.get());
        if (check_hip_mem_space == nullptr) {
            return false;
        }
        return true;
    }

    static void increase_num_execs(int device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]++;
    }

    static void decrease_num_execs(int device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        num_execs[device_id]--;
    }

    static int get_num_execs(int device_id)
    {
        std::lock_guard<std::mutex> guard(mutex[device_id]);
        return num_execs[device_id];
    }

private:
    int device_id_;
    std::shared_ptr<Executor> master_;
    int num_multiprocessor_;
    int num_warps_per_sm_;
    int major_;
    int minor_;
    int warp_size_;
    std::unique_ptr<hip_exec_info> exec_info_;
    std::shared_ptr<MemorySpace> mem_space_instance_;

    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T *)>>;
    handle_manager<hipblasContext> hipblas_handle_;
    handle_manager<hipsparseContext> hipsparse_handle_;

    static constexpr int max_devices = 64;
    static int num_execs[max_devices];
    static std::mutex mutex[max_devices];
};


namespace kernels {
namespace hip {
using DefaultExecutor = HipExecutor;
}  // namespace hip
}  // namespace kernels


#undef GKO_OVERRIDE_RAW_COPY_TO


}  // namespace gko


#endif  // GKO_CORE_BASE_EXECUTOR_HPP_
