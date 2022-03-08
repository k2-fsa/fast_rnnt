/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
 *
 * @copyright
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

#include "fast_rnnt/csrc/device_guard.h"
#include "fast_rnnt/csrc/utils.h"
#include "fast_rnnt/python/csrc/utils.h"

namespace fast_rnnt {

void PybindUtils(py::module &m) {
  m.def("monotonic_lower_bound_", [](torch::Tensor &src) -> void {
    DeviceGuard guard(src.device());
    if (src.dim() == 1) {
      MonotonicLowerBound(src);
    } else if (src.dim() == 2) {
      int32_t dim0 = src.sizes()[0];
      for (int32_t i = 0; i < dim0; ++i) {
        auto sub = src.index({i, torch::indexing::Slice()});
        MonotonicLowerBound(sub);
      }
    } else {
      TORCH_CHECK(false, "Only support 1 dimension and 2 dimensions tensor");
    }
  }, py::arg("src"));
}

} // namespace fast_rnnt
