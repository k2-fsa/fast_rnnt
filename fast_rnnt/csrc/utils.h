/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FAST_RNNT_CSRC_UTILS_H_
#define FAST_RNNT_CSRC_UTILS_H_

#include "torch/extension.h"

#ifdef FT_WITH_CUDA
#include "cub/cub.cuh" // NOLINT

namespace fast_rnnt {
namespace internal {

template <typename T> struct MinOp {
  __host__ __device__ __forceinline__ T operator()(const T &a,
                                                   const T &b) const {
    return (a > b) ? b : a;
  }
};

// Will be used (as both InputIterator and OutputIterator) in
// MonotonicLowerBound to call cub::DeviceScan::InclusiveScan.
template <typename T> struct ConstReversedPtr {
  const T *data;

  // data points to the last element now
  explicit ConstReversedPtr(const T *data, int32_t size)
      : data(data + size - 1) {}

  // operator[], operator+, and operator* are required by
  // cub::DeviceScan::InclusiveScan
  __host__ __device__ __forceinline__ const T &operator[](int32_t i) const {
    return data[-i];
  }
  __host__ __device__ __forceinline__ ConstReversedPtr
  operator+(int32_t n) const {
    ConstReversedPtr tmp(*this);
    tmp.data -= n;
    return tmp;
  }
  __host__ __device__ __forceinline__ const T &operator*() const {
    return *data;
  }
};

template <typename T> struct ReversedPtr {
  T *data;

  // data points to the last element now
  explicit ReversedPtr(T *data, int32_t size) : data(data + size - 1) {}

  // operator[], operator+, and operator* are required by
  // cub::DeviceScan::InclusiveScan
  __host__ __device__ __forceinline__ T &operator[](int32_t i) {
    return data[-i];
  }
  __host__ __device__ __forceinline__ ReversedPtr operator+(int32_t n) const {
    ReversedPtr tmp(*this);
    tmp.data -= n;
    return tmp;
  }
  __host__ __device__ __forceinline__ T &operator*() { return *data; }
};

} // namespace internal
} // namespace fast_rnnt

namespace std {
// vaule_type is required by cub::DeviceScan::InclusiveSum
template <typename T>
struct iterator_traits<fast_rnnt::internal::ConstReversedPtr<T>> {
  typedef T value_type;
};
template <typename T>
struct iterator_traits<fast_rnnt::internal::ReversedPtr<T>> {
  typedef T value_type;
};
} // namespace std
#endif   // FT_WITH_CUDA

namespace fast_rnnt {
void MonotonicLowerBound(torch::Tensor &src);
} // namespace fast_rnnt

#endif // FAST_RNNT_CSRC_UTILS_H_
