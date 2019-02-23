/**
 * @file   subarray.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017-2018 TileDB, Inc.
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
 * This file implements class Subarray.
 */

#include "tiledb/sm/subarray/subarray.h"
#include "tiledb/sm/array/array.h"
#include "tiledb/sm/misc/parallel_functions.h"
#include "tiledb/sm/rtree/rtree.h"

#include <iomanip>

/* ****************************** */
/*             MACROS             */
/* ****************************** */

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

namespace tiledb {
namespace sm {

/* ****************************** */
/*   CONSTRUCTORS & DESTRUCTORS   */
/* ****************************** */

Subarray::Subarray() {
  array_ = nullptr;
  layout_ = Layout::UNORDERED;
  result_est_size_computed_ = false;
  tile_overlap_computed_ = false;
}

Subarray::Subarray(const Array* array, Layout layout)
    : array_(array)
    , layout_(layout) {
  auto dim_num = array->array_schema()->dim_num();
  auto domain_type = array->array_schema()->domain()->type();
  for (uint32_t i = 0; i < dim_num; ++i)
    ranges_.emplace_back(domain_type);
  result_est_size_computed_ = false;
  tile_overlap_computed_ = false;
  add_default_ranges();
}

Subarray::Subarray(const Subarray& subarray)
    : Subarray() {
  // Make a deep-copy clone
  auto clone = subarray.clone();
  // Swap with the clone
  swap(clone);
}

Subarray::Subarray(Subarray&& subarray)
    : Subarray() {
  // Swap with the argument
  swap(subarray);
}

Subarray& Subarray::operator=(const Subarray& subarray) {
  // Make a deep-copy clone
  auto clone = subarray.clone();
  // Swap with the clone
  swap(clone);

  return *this;
}

Subarray& Subarray::operator=(Subarray&& subarray) {
  // Swap with the argument
  swap(subarray);

  return *this;
}

/* ****************************** */
/*               API              */
/* ****************************** */

Status Subarray::add_range(uint32_t dim_idx, const void* range) {
  if (range == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot add range to dimension; Range cannot be null"));

  auto dim_num = array_->array_schema()->dim_num();
  if (dim_idx >= dim_num)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot add range to dimension; Invalid dimension index"));

  auto type = array_->array_schema()->domain()->type();
  switch (type) {
    case Datatype::INT8:
      return add_range<int8_t>(dim_idx, (const int8_t*)range);
    case Datatype::UINT8:
      return add_range<uint8_t>(dim_idx, (const uint8_t*)range);
    case Datatype::INT16:
      return add_range<int16_t>(dim_idx, (const int16_t*)range);
    case Datatype::UINT16:
      return add_range<uint16_t>(dim_idx, (const uint16_t*)range);
    case Datatype::INT32:
      return add_range<int32_t>(dim_idx, (const int32_t*)range);
    case Datatype::UINT32:
      return add_range<uint32_t>(dim_idx, (const uint32_t*)range);
    case Datatype::INT64:
      return add_range<int64_t>(dim_idx, (const int64_t*)range);
    case Datatype::UINT64:
      return add_range<uint64_t>(dim_idx, (const uint64_t*)range);
    case Datatype::FLOAT32:
      return add_range<float>(dim_idx, (const float*)range);
    case Datatype::FLOAT64:
      return add_range<double>(dim_idx, (const double*)range);
    default:
      return LOG_STATUS(Status::SubarrayError(
          "Cannot add range to dimension; Unsupported subarray domain type"));
  }

  return Status::Ok();
}

const Array* Subarray::array() const {
  return array_;
}

template <class T>
uint64_t Subarray::cell_num(uint64_t range_idx) const {
  // Special case if it unary
  if (is_unary(range_idx))
    return 1;

  // Inapplicable to non-unary real ranges
  if (!std::is_integral<T>::value)
    return UINT64_MAX;

  uint64_t ret = 1, length;
  auto range = this->range<T>(range_idx);

  for (const auto& r : range) {
    // The code below essentially computes
    // ret *= r[1] - r[0] + 1;
    // while performing overflow checks
    length = r[1] - r[0];
    if (length == UINT64_MAX)  // overflow
      return UINT64_MAX;
    ++length;
    ret = utils::math::safe_mul(length, ret);
  }

  return ret;
}

void Subarray::clear() {
  ranges_.clear();
  range_offsets_.clear();
  tile_overlap_.clear();
  result_est_size_computed_ = false;
  tile_overlap_computed_ = false;
}

Status Subarray::compute_tile_overlap() {
  auto type = array_->array_schema()->domain()->type();
  switch (type) {
    case Datatype::INT8:
      return compute_tile_overlap<int8_t>();
    case Datatype::UINT8:
      return compute_tile_overlap<uint8_t>();
    case Datatype::INT16:
      return compute_tile_overlap<int16_t>();
    case Datatype::UINT16:
      return compute_tile_overlap<uint16_t>();
    case Datatype::INT32:
      return compute_tile_overlap<int32_t>();
    case Datatype::UINT32:
      return compute_tile_overlap<uint32_t>();
    case Datatype::INT64:
      return compute_tile_overlap<int64_t>();
    case Datatype::UINT64:
      return compute_tile_overlap<uint64_t>();
    case Datatype::FLOAT32:
      return compute_tile_overlap<float>();
    case Datatype::FLOAT64:
      return compute_tile_overlap<double>();
    default:
      return LOG_STATUS(Status::SubarrayError(
          "Failed to compute tile overlap; unsupported domain type"));
  }

  return Status::Ok();
}

uint32_t Subarray::dim_num() const {
  return array_->array_schema()->dim_num();
}

const void* Subarray::domain() const {
  return array_->array_schema()->domain()->domain();
}

bool Subarray::empty() const {
  return range_num() == 0;
}

Status Subarray::get_query_type(QueryType* type) const {
  if (array_ == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get query type from array; Invalid array"));

  return array_->get_query_type(type);
}

Status Subarray::get_range(
    uint32_t dim_idx, uint64_t range_idx, const void** range) const {
  auto dim_num = array_->array_schema()->dim_num();
  if (dim_idx >= dim_num)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get range; Invalid dimension index"));

  auto range_num = ranges_[dim_idx].range_num();
  if (range_idx >= range_num)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get range; Invalid range index"));

  *range = ranges_[dim_idx].get_range(range_idx);

  return Status::Ok();
}

Status Subarray::get_range_num(uint32_t dim_idx, uint64_t* range_num) const {
  auto dim_num = array_->array_schema()->dim_num();
  if (dim_idx >= dim_num)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get number of ranges for a dimension; "
                              "Invalid dimension index"));

  *range_num = ranges_[dim_idx].range_num();

  return Status::Ok();
}

template <class T>
Subarray Subarray::get_subarray(uint64_t start, uint64_t end) const {
  Subarray ret(array_, layout_);

  auto start_coords = get_range_coords(start);
  auto end_coords = get_range_coords(end);

  auto dim_num = this->dim_num();
  for (unsigned i = 0; i < dim_num; ++i) {
    for (uint64_t r = start_coords[i]; r <= end_coords[i]; ++r) {
      ret.add_range(i, ranges_[i].get_range(r));
    }
  }

  // Set tile overlap
  auto fragment_num = tile_overlap_.size();
  ret.tile_overlap_.resize(fragment_num);
  for (unsigned i = 0; i < fragment_num; ++i) {
    for (uint64_t r = start; r <= end; ++r) {
      ret.tile_overlap_[i].push_back(tile_overlap_[i][r]);
    }
  }

  // Compute range offsets
  ret.compute_range_offsets();

  return ret;
}

bool Subarray::is_unary() const {
  if (range_num() != 1)
    return false;

  for (const auto& range : ranges_) {
    auto r = (const uint8_t*)range.get_range(0);
    auto range_size = range.range_size_;
    if (std::memcmp(r, r + range_size / 2, range_size / 2) != 0)
      return false;
  }

  return true;
}

bool Subarray::is_unary(uint64_t range_idx) const {
  auto coords = get_range_coords(range_idx);
  auto dim_num = this->dim_num();

  for (unsigned i = 0; i < dim_num; ++i) {
    auto r = (const uint8_t*)ranges_[i].get_range(coords[i]);
    auto range_size = ranges_[i].range_size_;
    if (std::memcmp(r, r + range_size / 2, range_size / 2) != 0)
      return false;
  }

  return true;
}

Layout Subarray::layout() const {
  return layout_;
}

Status Subarray::get_est_result_size(const char* attr_name, uint64_t* size) {
  if (array_->array_schema()->dense())
    return LOG_STATUS(
        Status::SubarrayError("Cannot get estimated result size; Feature not "
                              "supported for dense arrays yet"));

  // Check attribute name
  if (attr_name == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Invalid attribute"));

  // Check attribute
  auto attr = array_->array_schema()->attribute(attr_name);
  if (attr_name != constants::coords && attr == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Invalid attribute"));

  // Check size pointer
  if (size == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Invalid size input"));

  // Check if the attribute is fixed-sized
  if (attr_name != constants::coords && attr->var_size())
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Attribute must be fixed-sized"));

  // Compute tile overlap for each fragment
  RETURN_NOT_OK(compute_est_result_size());
  *size = (uint64_t)ceil(est_result_size_[attr_name].size_fixed_);

  return Status::Ok();
}

Status Subarray::get_est_result_size(
    const char* attr_name, uint64_t* size_off, uint64_t* size_val) {
  if (array_->array_schema()->dense())
    return LOG_STATUS(
        Status::SubarrayError("Cannot get estimated result size; Feature not "
                              "supported for dense arrays yet"));

  // Check attribute name
  if (attr_name == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Invalid attribute"));

  // Check attribute
  auto attr = array_->array_schema()->attribute(attr_name);
  if (attr == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Invalid attribute"));

  // Check size pointer
  if (size_off == nullptr || size_val == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Invalid size input"));

  // Check if the attribute is var-sized
  if (!attr->var_size())
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get estimated result size; Attribute must be var-sized"));

  // Compute tile overlap for each fragment
  RETURN_NOT_OK(compute_est_result_size());
  *size_off = (uint64_t)ceil(est_result_size_[attr_name].size_fixed_);
  *size_val = (uint64_t)ceil(est_result_size_[attr_name].size_var_);

  return Status::Ok();
}

Status Subarray::get_max_memory_size(const char* attr_name, uint64_t* size) {
  if (array_->array_schema()->dense())
    return LOG_STATUS(
        Status::SubarrayError("Cannot get maximum memory size; Feature not "
                              "supported for dense arrays yet"));

  // Check attribute name
  if (attr_name == nullptr)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get max memory size; Invalid attribute"));

  // Check attribute
  auto attr = array_->array_schema()->attribute(attr_name);
  if (attr_name != constants::coords && attr == nullptr)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get max memory size; Invalid attribute"));

  // Check size pointer
  if (size == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get max memory size; Invalid size input"));

  // Check if the attribute is fixed-sized
  if (attr_name != constants::coords && attr->var_size())
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get max memory size; Attribute must be fixed-sized"));

  // Compute tile overlap for each fragment
  compute_est_result_size();
  *size = (uint64_t)ceil(est_result_size_[attr_name].mem_size_fixed_);

  return Status::Ok();
}

Status Subarray::get_max_memory_size(
    const char* attr_name, uint64_t* size_off, uint64_t* size_val) {
  if (array_->array_schema()->dense())
    return LOG_STATUS(
        Status::SubarrayError("Cannot get max memory size; Feature not "
                              "supported for dense arrays yet"));

  // Check attribute name
  if (attr_name == nullptr)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get max memory size; Invalid attribute"));

  // Check attribute
  auto attr = array_->array_schema()->attribute(attr_name);
  if (attr == nullptr)
    return LOG_STATUS(
        Status::SubarrayError("Cannot get max memory size; Invalid attribute"));

  // Check size pointer
  if (size_off == nullptr || size_val == nullptr)
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get max memory size; Invalid size input"));

  // Check if the attribute is var-sized
  if (!attr->var_size())
    return LOG_STATUS(Status::SubarrayError(
        "Cannot get max memory size; Attribute must be var-sized"));

  // Compute tile overlap for each fragment
  compute_est_result_size();
  *size_off = (uint64_t)ceil(est_result_size_[attr_name].mem_size_fixed_);
  *size_val = (uint64_t)ceil(est_result_size_[attr_name].mem_size_var_);

  return Status::Ok();
}

std::vector<uint64_t> Subarray::get_range_coords(uint64_t range_idx) const {
  std::vector<uint64_t> ret;

  uint64_t tmp_idx = range_idx;
  auto dim_num = this->dim_num();
  auto cell_order = array_->array_schema()->cell_order();
  auto layout = (layout_ == Layout::UNORDERED) ? cell_order : layout_;

  if (layout == Layout::ROW_MAJOR) {
    for (unsigned i = 0; i < dim_num; ++i) {
      ret.push_back(tmp_idx / range_offsets_[i]);
      tmp_idx %= range_offsets_[i];
    }
  } else if (layout == Layout::COL_MAJOR) {
    for (unsigned i = dim_num - 1;; --i) {
      ret.push_back(tmp_idx / range_offsets_[i]);
      tmp_idx %= range_offsets_[i];
      if (i == 0)
        break;
    }
    std::reverse(ret.begin(), ret.end());
  } else {
    assert(false);
  }

  return ret;
}

uint64_t Subarray::range_idx(const std::vector<uint64_t>& range_coords) const {
  uint64_t ret = 0;
  auto dim_num = this->dim_num();
  for (unsigned i = 0; i < dim_num; ++i)
    ret += range_offsets_[i] * range_coords[i];

  return ret;
}

uint64_t Subarray::range_num() const {
  if (ranges_.empty())
    return 0;

  uint64_t ret = 1;
  for (const auto& r : ranges_)
    ret *= r.range_num();

  return ret;
}

template <class T>
std::vector<const T*> Subarray::range(uint64_t range_idx) const {
  std::vector<const T*> ret;
  uint64_t tmp_idx = range_idx;
  auto dim_num = this->dim_num();
  auto cell_order = array_->array_schema()->cell_order();
  auto layout = (layout_ == Layout::UNORDERED) ? cell_order : layout_;

  if (layout == Layout::ROW_MAJOR) {
    for (unsigned i = 0; i < dim_num; ++i) {
      ret.push_back((T*)(ranges_[i].get_range(tmp_idx / range_offsets_[i])));
      tmp_idx %= range_offsets_[i];
    }
  } else if (layout == Layout::COL_MAJOR) {
    for (unsigned i = dim_num - 1;; --i) {
      ret.push_back((T*)(ranges_[i].get_range(tmp_idx / range_offsets_[i])));
      tmp_idx %= range_offsets_[i];
      if (i == 0)
        break;
    }
    std::reverse(ret.begin(), ret.end());
  } else {
    assert(false);
  }

  return ret;
}

const std::vector<std::vector<TileOverlap>>& Subarray::tile_overlap() const {
  return tile_overlap_;
}

Datatype Subarray::type() const {
  return array_->array_schema()->domain()->type();
}

/* ****************************** */
/*          PRIVATE METHODS       */
/* ****************************** */

void Subarray::add_default_ranges() {
  auto dim_num = array_->array_schema()->dim_num();
  auto domain = (unsigned char*)array_->array_schema()->domain()->domain();
  for (unsigned i = 0; i < dim_num; ++i) {
    auto range_size = ranges_[i].range_size_;
    ranges_[i].add_range(&(domain[i * range_size]), true);
  }
}

template <class T>
Status Subarray::add_range(uint32_t dim_idx, const T* range) {
  assert(dim_idx < array_->array_schema()->dim_num());

  // Must reset the result size and tile overlap
  result_est_size_computed_ = false;
  tile_overlap_computed_ = false;

  // Check for NaN
  RETURN_NOT_OK(check_nan<T>(range));

  // Check range bounds
  if (range[0] > range[1])
    return LOG_STATUS(
        Status::SubarrayError("Cannot add range to dimension; Lower range "
                              "bound cannot be larger than the higher bound"));

  // Check range against the domain
  auto domain = (T*)array_->array_schema()->domain()->domain();
  if (range[0] < domain[2 * dim_idx] || range[1] > domain[2 * dim_idx + 1])
    return LOG_STATUS(
        Status::SubarrayError("Cannot add range to dimension; Range must be in "
                              "the domain the subarray is constructed from"));

  // Add the range
  ranges_[dim_idx].add_range(range);

  return Status::Ok();
}

void Subarray::compute_range_offsets() {
  range_offsets_.clear();

  auto dim_num = this->dim_num();
  auto cell_order = array_->array_schema()->cell_order();
  auto layout = (layout_ == Layout::UNORDERED) ? cell_order : layout_;

  if (layout == Layout::COL_MAJOR) {
    range_offsets_.push_back(1);
    if (dim_num > 1) {
      for (unsigned int i = 1; i < dim_num; ++i)
        range_offsets_.push_back(
            range_offsets_.back() * ranges_[i - 1].range_num());
    }
  } else if (layout == Layout::ROW_MAJOR) {
    range_offsets_.push_back(1);
    if (dim_num > 1) {
      for (unsigned int i = dim_num - 2;; --i) {
        range_offsets_.push_back(
            range_offsets_.back() * ranges_[i + 1].range_num());
        if (i == 0)
          break;
      }
    }
    std::reverse(range_offsets_.begin(), range_offsets_.end());
  } else {
    assert(false);
  }
}

Status Subarray::compute_est_result_size() {
  if (result_est_size_computed_)
    return Status::Ok();

  auto type = array_->array_schema()->domain()->type();
  switch (type) {
    case Datatype::INT8:
      return compute_est_result_size<int8_t>();
    case Datatype::UINT8:
      return compute_est_result_size<uint8_t>();
    case Datatype::INT16:
      return compute_est_result_size<int16_t>();
    case Datatype::UINT16:
      return compute_est_result_size<uint16_t>();
    case Datatype::INT32:
      return compute_est_result_size<int32_t>();
    case Datatype::UINT32:
      return compute_est_result_size<uint32_t>();
    case Datatype::INT64:
      return compute_est_result_size<int64_t>();
    case Datatype::UINT64:
      return compute_est_result_size<uint64_t>();
    case Datatype::FLOAT32:
      return compute_est_result_size<float>();
    case Datatype::FLOAT64:
      return compute_est_result_size<double>();
    default:
      return LOG_STATUS(Status::SubarrayError(
          "Cannot compute estimated results size; unsupported domain type"));
  }

  return Status::Ok();
}

template <class T>
Status Subarray::compute_est_result_size() {
  if (result_est_size_computed_)
    return Status::Ok();

  RETURN_NOT_OK(compute_tile_overlap<T>());

  std::mutex mtx;

  // Prepare estimated result size vector for all attributes and coords
  auto attributes = array_->array_schema()->attributes();
  auto attribute_num = attributes.size();
  std::vector<ResultSize> est_result_size_vec;
  for (unsigned i = 0; i < attribute_num + 1; ++i)
    est_result_size_vec.emplace_back(ResultSize{0.0, 0.0, 0, 0});

  // Compute estimated result in parallel over fragments and ranges
  auto meta = array_->fragment_metadata();
  auto fragment_num = meta.size();
  tile_overlap_.resize(fragment_num);
  auto range_num = this->range_num();

  auto statuses_1 = parallel_for(0, range_num, [&](uint64_t i) {
    for (unsigned a = 0; a < attribute_num + 1; ++a) {
      auto attr_name =
          (a == attribute_num) ? constants::coords : attributes[a]->name();
      bool var_size = (a == attribute_num) ? false : attributes[a]->var_size();
      auto result_size = compute_est_result_size<T>(attr_name, i, var_size);
      std::lock_guard<std::mutex> block(mtx);
      est_result_size_vec[a].size_fixed_ += result_size.size_fixed_;
      est_result_size_vec[a].size_var_ += result_size.size_var_;
      est_result_size_vec[a].mem_size_fixed_ += result_size.mem_size_fixed_;
      est_result_size_vec[a].mem_size_var_ += result_size.mem_size_var_;
    }
    return Status::Ok();
  });

  // Amplify result estimation
  if (constants::est_result_size_amplification != 1.0) {
    for (auto& r : est_result_size_vec) {
      r.size_fixed_ *= constants::est_result_size_amplification;
      r.size_var_ *= constants::est_result_size_amplification;
    }
  }

  // Set the estimated result size map
  est_result_size_.clear();
  for (unsigned a = 0; a < attribute_num + 1; ++a) {
    auto attr_name =
        (a == attribute_num) ? constants::coords : attributes[a]->name();
    est_result_size_[attr_name] = est_result_size_vec[a];
  }
  result_est_size_computed_ = true;

  return Status::Ok();
}

template <class T>
Subarray::ResultSize Subarray::compute_est_result_size(
    const std::string& attr_name, uint64_t range_idx, bool var_size) const {
  // For easy reference
  auto fragment_num = array_->fragment_metadata().size();
  ResultSize ret{0.0, 0.0, 0, 0};
  auto array_schema = array_->array_schema();

  // Compute estimated result
  for (unsigned f = 0; f < fragment_num; ++f) {
    const auto& overlap = tile_overlap_[f][range_idx];
    auto meta = array_->fragment_metadata()[f];

    // Parse tile ranges
    for (const auto& tr : overlap.tile_ranges_) {
      for (uint64_t tid = tr.first; tid <= tr.second; ++tid) {
        if (!var_size) {
          ret.size_fixed_ += meta->tile_size(attr_name, tid);
          ret.mem_size_fixed_ += meta->tile_size(attr_name, tid);
        } else {
          ret.size_fixed_ += meta->tile_size(attr_name, tid);
          ret.size_var_ += meta->tile_var_size(attr_name, tid);
          ret.mem_size_fixed_ += meta->tile_size(attr_name, tid);
          ret.mem_size_var_ += meta->tile_var_size(attr_name, tid);
        }
      }
    }

    // Parse individual tiles
    for (const auto& t : overlap.tiles_) {
      auto tid = t.first;
      auto ratio = t.second;
      if (!var_size) {
        ret.size_fixed_ += meta->tile_size(attr_name, tid) * ratio;
        ret.mem_size_fixed_ += meta->tile_size(attr_name, tid);
      } else {
        ret.size_fixed_ += meta->tile_size(attr_name, tid) * ratio;
        ret.size_var_ += meta->tile_var_size(attr_name, tid) * ratio;
        ret.mem_size_fixed_ += meta->tile_size(attr_name, tid);
        ret.mem_size_var_ += meta->tile_var_size(attr_name, tid);
      }
    }
  }

  // Calibrate result
  uint64_t max_size_fixed, max_size_var = UINT64_MAX;
  auto cell_num = this->cell_num<T>(range_idx);
  if (var_size) {
    max_size_fixed =
        utils::math::safe_mul(cell_num, constants::cell_var_offset_size);
  } else {
    max_size_fixed =
        utils::math::safe_mul(cell_num, array_schema->cell_size(attr_name));
  }
  ret.size_fixed_ = MIN(ret.size_fixed_, max_size_fixed);
  ret.size_var_ = MIN(ret.size_var_, max_size_var);

  return ret;
}

template <class T>
Status Subarray::compute_tile_overlap() {
  if (tile_overlap_computed_)
    return Status::Ok();

  compute_range_offsets();
  tile_overlap_.clear();
  auto meta = array_->fragment_metadata();
  auto fragment_num = meta.size();
  tile_overlap_.resize(fragment_num);
  auto range_num = this->range_num();
  for (unsigned i = 0; i < fragment_num; ++i)
    tile_overlap_[i].resize(range_num);

  // Compute estimated tile overlap in parallel over fragments and ranges
  auto statuses = parallel_for_2d(
      0, fragment_num, 0, range_num, [&](unsigned i, unsigned j) {
        // TODO: may return error
        auto rtree = meta[i]->rtree();
        auto range = this->range<T>(j);
        tile_overlap_[i][j] = rtree->get_tile_overlap<T>(range);
        return Status::Ok();
      });

  tile_overlap_computed_ = true;

  return Status::Ok();
}

Subarray Subarray::clone() const {
  Subarray clone;
  clone.array_ = array_;
  clone.layout_ = layout_;
  clone.ranges_ = ranges_;
  clone.range_offsets_ = range_offsets_;
  clone.tile_overlap_ = tile_overlap_;
  clone.result_est_size_computed_ = result_est_size_computed_;
  clone.tile_overlap_computed_ = tile_overlap_computed_;

  return clone;
}

void Subarray::swap(Subarray& subarray) {
  std::swap(array_, subarray.array_);
  std::swap(layout_, subarray.layout_);
  std::swap(ranges_, subarray.ranges_);
  std::swap(range_offsets_, subarray.range_offsets_);
  std::swap(tile_overlap_, subarray.tile_overlap_);
  std::swap(result_est_size_computed_, subarray.result_est_size_computed_);
  std::swap(tile_overlap_computed_, subarray.tile_overlap_computed_);
}

// Explicit instantiations
template Subarray Subarray::get_subarray<uint8_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<int8_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<uint16_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<int16_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<uint32_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<int32_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<uint64_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<int64_t>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<float>(
    uint64_t start, uint64_t end) const;
template Subarray Subarray::get_subarray<double>(
    uint64_t start, uint64_t end) const;

template uint64_t Subarray::cell_num<int8_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<uint8_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<int16_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<uint16_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<int32_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<uint32_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<int64_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<uint64_t>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<float>(uint64_t range_idx) const;
template uint64_t Subarray::cell_num<double>(uint64_t range_idx) const;

}  // namespace sm
}  // namespace tiledb
