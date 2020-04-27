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

#ifndef GKO_HIP_BASE_EXECUTOR_HIP_HPP_
#define GKO_HIP_BASE_EXECUTOR_HIP_HPP_

#include <ginkgo/core/base/executor.hpp>


#include <hip/hip_runtime.h>


#include "hip/base/types.hip.hpp"


namespace gko {


template <typename Kernel, typename... Args>
void HipExecutor::run_gpu(const char *kernel_name, Kernel kernel,
                          size_type num_blocks, size_type block_size,
                          Args... args) const
{
    this->template log<log::Logger::gpu_kernel_launch>(this, kernel_name,
                                                       num_blocks, block_size);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel), num_blocks, block_size, 0, 0,
                       kernels::hip::as_hip_type(args)...);
    this->template log<log::Logger::gpu_kernel_finish>(this, kernel_name,
                                                       num_blocks, block_size);
}


}  // namespace gko

#endif  // GKO_HIP_BASE_EXECUTOR_HIP_HPP_