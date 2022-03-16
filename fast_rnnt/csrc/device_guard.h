/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang, Wei Kang)
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

#ifndef FAST_RNNT_CSRC_DEVICE_GUARD_H_
#define FAST_RNNT_CSRC_DEVICE_GUARD_H_

#include "torch/script.h"

// This file is modified from
// https://github.com/k2-fsa/k2/blob/master/k2/csrc/device_guard.h
namespace fast_rnnt {

// DeviceGuard is an RAII class. Its sole purpose is to restore
// the previous default cuda device if a CUDA context changes the
// current default cuda device.
class DeviceGuard {
public:
  explicit DeviceGuard(torch::Device device) {
    if (device.type() == torch::kCUDA) {
      old_device_ = GetDevice();
      new_device_ = device.index();
      if (old_device_ != new_device_)
        SetDevice(new_device_);
    }
    // else do nothing
  }

  explicit DeviceGuard(int32_t new_device) : new_device_(new_device) {
    if (new_device != -1) {
      old_device_ = GetDevice();
      if (old_device_ != new_device)
        SetDevice(new_device);
    }
  }

  ~DeviceGuard() {
    if (old_device_ != -1 && old_device_ != new_device_) {
      // restore the previous device
      SetDevice(old_device_);
    }
    // else it was either a CPU context or the device IDs
    // were the same
  }

  DeviceGuard(const DeviceGuard &) = delete;
  DeviceGuard &operator=(const DeviceGuard &) = delete;

  DeviceGuard(DeviceGuard &&) = delete;
  DeviceGuard &operator=(DeviceGuard &&) = delete;

private:
  static int32_t GetDevice() {
#ifdef FT_WITH_CUDA
    int32_t device;
    auto s = cudaGetDevice(&device);
    TORCH_CHECK(s == cudaSuccess, cudaGetErrorString(s));
    return device;
#else
    return -1;
#endif
  }

  static void SetDevice(int32_t device) {
#ifdef FT_WITH_CUDA
    auto s = cudaSetDevice(device);
    TORCH_CHECK(s == cudaSuccess, cudaGetErrorString(s));
#else
    return;
#endif
  }

private:
  int32_t old_device_ = -1;
  int32_t new_device_ = -1;
};

} // namespace fast_rnnt

#endif // FAST_RNNT_CSRC_DEVICE_GUARD_H_
