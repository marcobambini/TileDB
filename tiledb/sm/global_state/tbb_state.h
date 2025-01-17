/**
 * @file   tbb_state.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2018-2021 TileDB, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * @section DESCRIPTION
 *
 This file declares the Intel TBB threading scheduler state, if TileDB is built
 with Intel TBB.
 */

#ifndef TILEDB_TBB_STATE_H
#define TILEDB_TBB_STATE_H

#include "tiledb/common/status.h"

using namespace tiledb::common;

namespace tiledb {
namespace sm {

class Config;

namespace global_state {

/**
 * Initializes the Intel TBB thread runtime scheduler with a specified number of
 * threads
 *
 * @param config TileDB Config object pointer (or nullptr)
 * @return Status
 */
Status init_tbb(const Config* config);

}  // namespace global_state
}  // namespace sm
}  // namespace tiledb

#endif
