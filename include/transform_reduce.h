#pragma once

#include <vector>
#include <enum_quda.h>
#include <reducer.h>

/**
   @file transform_reduce.h

   @brief QUDA reimplementation of thrust::transform_reduce as well as
   wrappers also implementing thrust::reduce.
 */

namespace quda
{

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Vector of results
     @param[in] v Vector of inputs
     @param[in] n_items Number of elements to be reduced in each input
     @param[in] transformer Functor that applies transform to each element
     @param[in] mapper Index mapper (defaults to identity)
   */
  template <typename reducer, typename T, typename I, typename transformer, typename mapper = identity>
  void transform_reduce(QudaFieldLocation location, std::vector<typename reducer::reduce_t> &result,
                        const std::vector<T *> &v, I n_items, transformer h, mapper m = identity());

  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
     @param[in] transformer Functor that applies transform to each element
   */
  template <typename reducer, typename T, typename I, typename transformer>
  typename reducer::reduce_t transform_reduce(QudaFieldLocation location, const T *v, I n_items, transformer h);

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory, and is a batched implementation.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
   */
  template <typename reducer, typename T, typename I, typename mapper = identity>
  void reduce(QudaFieldLocation location, std::vector<typename reducer::reduce_t> &result, const std::vector<T *> &v,
              I n_items);

  /**
     @brief QUDA implementation providing thrust::reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] location Location where the reduction will take place
     @param[out] result Result
     @param[in] v Input vector
     @param[in] n_items Number of elements to be reduced
   */
  template <typename reducer, typename T, typename I>
  typename reducer::reduce_t reduce(QudaFieldLocation location, const T *v, I n_items);

} // namespace quda
