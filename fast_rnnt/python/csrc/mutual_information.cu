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
#include "fast_rnnt/csrc/mutual_information.h"
#include "fast_rnnt/python/csrc/mutual_information.h"

namespace fast_rnnt {
void PybindMutualInformation(py::module &m) {
  m.def(
      "mutual_information_forward",
      [](torch::Tensor px, torch::Tensor py,
         torch::optional<torch::Tensor> boundary,
         torch::Tensor p) -> torch::Tensor {
        fast_rnnt::DeviceGuard guard(px.device());
        if (px.device().is_cpu()) {
          return MutualInformationCpu(px, py, boundary, p);
        } else {
#ifdef FT_WITH_CUDA
          return MutualInformationCuda(px, py, boundary, p);
#else
          TORCH_CHECK(false, "Failed to find native CUDA module, make sure "
                             "that you compiled the code with K2_WITH_CUDA.");
          return torch::Tensor();
#endif
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"), py::arg("p"));

  m.def(
      "mutual_information_backward",
      [](torch::Tensor px, torch::Tensor py,
         torch::optional<torch::Tensor> boundary, torch::Tensor p,
         torch::Tensor ans_grad) -> std::vector<torch::Tensor> {
        fast_rnnt::DeviceGuard guard(px.device());
        if (px.device().is_cpu()) {
          return MutualInformationBackwardCpu(px, py, boundary, p, ans_grad);
        } else {
#ifdef FT_WITH_CUDA
          return MutualInformationBackwardCuda(px, py, boundary, p, ans_grad,
                                               true);
#else
          TORCH_CHECK(false, "Failed to find native CUDA module, make sure "
                             "that you compiled the code with K2_WITH_CUDA.");
          return std::vector<torch::Tensor>();
#endif
        }
      },
      py::arg("px"), py::arg("py"), py::arg("boundary"), py::arg("p"),
      py::arg("ans_grad"));
}
} // namespace fast_rnnt
