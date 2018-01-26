/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_KERNELS_HPP_


#include "core/matrix/csr.hpp"
#include "core/preconditioner/block_jacobi.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL(ValueType, IndexType)    \
    void find_blocks(const matrix::Csr<ValueType, IndexType> *system_matrix, \
                     uint32 max_block_size, size_type &num_blocks,           \
                     Array<IndexType> &block_pointers)

#define GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL(ValueType, IndexType)       \
    void generate(const matrix::Csr<ValueType, IndexType> *system_matrix,    \
                  size_type num_blocks, uint32 max_block_size,               \
                  size_type padding, const Array<IndexType> &block_pointers, \
                  Array<ValueType> &blocks)

#define DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL(ValueType, IndexType)


namespace cpu {
namespace block_jacobi {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace block_jacobi
}  // namespace cpu


namespace gpu {
namespace block_jacobi {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace block_jacobi
}  // namespace gpu


namespace reference {
namespace block_jacobi {

DECLARE_ALL_AS_TEMPLATES;

}  // namespace block_jacobi
}  // namespace reference


#undef DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BLOCK_JACOBI_KERNELS_HPP_