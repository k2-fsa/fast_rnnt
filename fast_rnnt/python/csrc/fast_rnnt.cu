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

#include "fast_rnnt/python/csrc/fast_rnnt.h"
#include "fast_rnnt/python/csrc/mutual_information.h"
#include "fast_rnnt/python/csrc/utils.h"

PYBIND11_MODULE(_fast_rnnt, m) {
  m.doc() = "Python wrapper for Fast Rnnt.";

  fast_rnnt::PybindMutualInformation(m);
  fast_rnnt::PybindUtils(m);
}
