/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2017 by Contributors
 * \file dlpackcpp.h
 * \brief Example C++ wrapper of DLPack
 */
#ifndef DLPACK_DLPACKCPP_H_
#define DLPACK_DLPACKCPP_H_

#include <dlpack/dlpack.h>

#include <cstdint>  // for int64_t etc
#include <cstdlib>  // for free()
#include <functional>  // for std::multiplies
#include <memory>
#include <numeric>
#include <vector>

namespace dlpack {

// Example container wrapping of DLTensor.
class DLTContainer {
 public:
  DLTContainer() {
    // default to float32
    handle_.data = nullptr;
    handle_.dtype.code = kDLFloat;
    handle_.dtype.bits = 32U;
    handle_.dtype.lanes = 1U;
    handle_.ctx.device_type = kDLCPU;
    handle_.ctx.device_id = 0;
    handle_.shape = nullptr;
    handle_.strides = nullptr;
    handle_.byte_offset = 0;
  }
  ~DLTContainer() {
    if (origin_ == nullptr) {
      free(handle_.data);
    }
  }
  operator DLTensor() {
    return handle_;
  }
  operator DLTensor*() {
    return &(handle_);
  }
  void Reshape(const std::vector<int64_t>& shape) {
    shape_ = shape;
    int64_t sz = std::accumulate(std::begin(shape), std::end(shape),
                                 int64_t(1), std::multiplies<int64_t>());
    int ret = posix_memalign(&handle_.data, 256, sz);
    if (ret != 0) throw std::bad_alloc();
    handle_.shape = &shape_[0];
    handle_.ndim = static_cast<uint32_t>(shape.size());
  }

 private:
  DLTensor handle_;
  std::vector<int64_t> shape_;
  std::vector<int64_t> strides_;
  // original space container, if
  std::shared_ptr<DLTContainer> origin_;
};

}  // namespace dlpack
#endif  // DLPACK_DLPACKCPP_H_
