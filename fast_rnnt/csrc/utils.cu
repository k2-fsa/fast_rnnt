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

#include "fast_rnnt/csrc/utils.h"

namespace fast_rnnt  {

void MonotonicLowerBound(torch::Tensor &src) {
  TORCH_CHECK(src.dim() == 1, "Only support one dimension tensor");
  TORCH_CHECK(src.scalar_type() == torch::kLong, "Only support LongTensor");
  TORCH_CHECK(src.is_contiguous(), "Expected to be contiguous");
  int32_t dim = src.numel();
  if (src.device().type() == torch::kCPU) {
    int64_t min_value = std::numeric_limits<int64_t>::max();
    int64_t *src_data = src.data_ptr<int64_t>();
    for (int32_t i = dim - 1; i >= 0; --i) {
      min_value = std::min(src_data[i], min_value);
      src[i] = min_value;
    }
  } else {
#ifdef FT_WITH_CUDA
    TORCH_CHECK(src.device().is_cuda());
    internal::MinOp<int64_t> min_op;
    auto src_data = src.data_ptr<int64_t>();
    internal::ConstReversedPtr<int64_t> src_ptr =
        internal::ConstReversedPtr<int64_t>(src_data, dim);
    internal::ReversedPtr<int64_t> dest_ptr =
        internal::ReversedPtr<int64_t>(src_data, dim);
    // The first time is to determine temporary device storage requirements.
    std::size_t temp_storage_bytes = 0;
    auto s = cub::DeviceScan::InclusiveScan(nullptr, temp_storage_bytes,
                                            src_ptr, dest_ptr, min_op, dim);
    TORCH_CHECK(s == cudaSuccess, cudaGetErrorString(s));

    auto d_temp = torch::empty({static_cast<int64_t>(temp_storage_bytes)},
                               torch::dtype(torch::kInt8).device(src.device()));
    int8_t *d_temp_storage = d_temp.data_ptr<int8_t>();
    s = cub::DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes,
                                       src_ptr, dest_ptr, min_op, dim);
    TORCH_CHECK(s == cudaSuccess, cudaGetErrorString(s));
#else
    TORCH_CHECK(false, "Please build with -DFT_WITH_CUDA=ON");
#endif  // FT_WITH_CUDA
  }
}

} // namespace fast_rnnt

