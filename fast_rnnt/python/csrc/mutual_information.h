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

#ifndef FAST_RNNT_PYTHON_CSRC_MUTUAL_INFORMATION_H_
#define FAST_RNNT_PYTHON_CSRC_MUTUAL_INFORMATION_H_

#include "fast_rnnt/python/csrc/fast_rnnt.h"

namespace fast_rnnt {

void PybindMutualInformation(py::module &m);

} // namespace fast_rnnt

#endif // FAST_RNNT_PYTHON_CSRC_MUTUAL_INFORMATION_H_
