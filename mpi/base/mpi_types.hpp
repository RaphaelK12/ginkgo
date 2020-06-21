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

#ifndef GKO_MPI_TYPES_HPP_
#define GKO_MPI_TYPES_HPP_


#include <mpi.h>

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
/**
 * @brief The bindings namespace.
 *
 * @ingroup bindings
 */
namespace helpers {
/**
 * @brief The MPI namespace.
 *
 * @ingroup mpi
 */
namespace mpi {

#define GKO_MPI_DATATYPE(BaseType, MPIType)                                  \
    inline MPI_Datatype get_mpi_type(const BaseType &) { return MPIType; }   \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


GKO_MPI_DATATYPE(char, MPI_CHAR);
GKO_MPI_DATATYPE(unsigned char, MPI_UNSIGNED_CHAR);
GKO_MPI_DATATYPE(unsigned, MPI_UNSIGNED);
GKO_MPI_DATATYPE(int, MPI_INT);
GKO_MPI_DATATYPE(unsigned long, MPI_UNSIGNED_LONG);
GKO_MPI_DATATYPE(unsigned short, MPI_UNSIGNED_SHORT);
GKO_MPI_DATATYPE(long, MPI_LONG);
GKO_MPI_DATATYPE(float, MPI_FLOAT);
GKO_MPI_DATATYPE(double, MPI_DOUBLE);
GKO_MPI_DATATYPE(long double, MPI_LONG_DOUBLE);
GKO_MPI_DATATYPE(std::complex<float>, MPI_COMPLEX);
GKO_MPI_DATATYPE(std::complex<double>, MPI_DOUBLE_COMPLEX);


}  // namespace mpi
}  // namespace helpers
}  // namespace gko


#endif  // GKO_MPI_BINDINGS_HPP_
