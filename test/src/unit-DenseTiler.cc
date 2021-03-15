/**
 * @file unit-DenseTiler.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017-2021 TileDB, Inc.
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
 * Tests the `DenseTiler` class.
 */

#include "tiledb/sm/c_api/tiledb.h"
#include "tiledb/sm/c_api/tiledb_struct_def.h"
#include "tiledb/sm/cpp_api/tiledb"
#include "tiledb/sm/query/dense_tiler.h"

#include <catch.hpp>
#include <iostream>

using namespace tiledb;
using namespace tiledb::sm;

struct DenseTilerFx {
  // Constants
  const int32_t fill_value = 0;

  // Aux structs
  struct DimensionInfo {
    std::string name;
    tiledb_datatype_t type;
    void* domain;
    void* tile_extent;
  };
  struct AttributeInfo {
    std::string name;
    tiledb_datatype_t type;
    uint32_t cell_val_num;
  };

  // Members used to create a subarray
  tiledb_ctx_t* ctx_;
  tiledb_array_t* array_;

  // Constructors/Destructors
  DenseTilerFx();
  ~DenseTilerFx();

  // Aux functions
  void remove_array(const std::string& array_name);
  void create_array(
      const std::string& array_name,
      const std::vector<DimensionInfo>& dim_info,
      const std::vector<AttributeInfo>& attr_info,
      tiledb_layout_t cell_order,
      tiledb_layout_t tile_order);
  void open_array(const std::string& array_name, tiledb_query_type_t type);
  void close_array();
  void add_ranges(
      const std::vector<const void*>& ranges,
      uint64_t range_size,
      Subarray* subarray);
  template <class T>
  bool check_tile(Tile* tile, const std::vector<T>& data);
};

DenseTilerFx::DenseTilerFx() {
  REQUIRE(tiledb_ctx_alloc(NULL, &ctx_) == TILEDB_OK);
  array_ = NULL;
}

DenseTilerFx::~DenseTilerFx() {
  close_array();
  tiledb_array_free(&array_);
  tiledb_ctx_free(&ctx_);
}

void DenseTilerFx::remove_array(const std::string& array_name) {
  tiledb::Context ctx;
  tiledb::VFS vfs(ctx);
  if (vfs.is_dir(array_name))
    vfs.remove_dir(array_name);
}

void DenseTilerFx::create_array(
    const std::string& array_name,
    const std::vector<DimensionInfo>& dim_info,
    const std::vector<AttributeInfo>& attr_info,
    tiledb_layout_t cell_order,
    tiledb_layout_t tile_order) {
  tiledb::Context ctx;

  // Clean array if it exists
  remove_array(array_name);

  // Create domain
  tiledb::Domain domain(ctx);
  for (const auto& di : dim_info) {
    auto d = tiledb::Dimension::create(
        ctx, di.name, di.type, di.domain, di.tile_extent);
    domain.add_dimension(d);
  }

  // Create array schema
  tiledb::ArraySchema schema(ctx, TILEDB_DENSE);
  schema.set_domain(domain);
  schema.set_cell_order(cell_order);
  schema.set_tile_order(tile_order);

  // Create attributes
  for (const auto& ai : attr_info) {
    auto a = tiledb::Attribute::create(ctx, ai.name, ai.type);
    a.set_cell_val_num(ai.cell_val_num);
    schema.add_attribute(a);
  }

  // Create array
  tiledb::Array::create(array_name, schema);
}

void DenseTilerFx::add_ranges(
    const std::vector<const void*>& ranges,
    uint64_t range_size,
    Subarray* subarray) {
  for (size_t i = 0; i < ranges.size(); ++i)
    CHECK(subarray->add_range(i, Range(ranges[i], range_size)).ok());
}

void DenseTilerFx::open_array(
    const std::string& array_name, tiledb_query_type_t type) {
  close_array();
  REQUIRE(tiledb_array_alloc(ctx_, array_name.c_str(), &array_) == TILEDB_OK);
  REQUIRE(tiledb_array_open(ctx_, array_, type) == TILEDB_OK);
}

void DenseTilerFx::close_array() {
  if (array_ == NULL)
    return;

  int32_t is_open;
  REQUIRE(tiledb_array_is_open(ctx_, array_, &is_open) == TILEDB_OK);
  if (!is_open)
    return;

  REQUIRE(tiledb_array_close(ctx_, array_) == TILEDB_OK);
  tiledb_array_free(&array_);
  array_ = NULL;
}

template <class T>
bool DenseTilerFx::check_tile(Tile* tile, const std::vector<T>& data) {
  std::vector<T> tile_data(data.size());
  CHECK(tile->read(&tile_data[0], data.size() * sizeof(T)).ok());
  CHECK(tile_data == data);
  return tile_data == data;
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test initialization, 1D",
    "[DenseTiler][init][1d]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {1, 10};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {3, 6};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of initialization
  CHECK(tiler1.tile_num() == 2);
  CHECK(tiler1.first_sub_tile_coords() == std::vector<uint64_t>{0});
  CHECK(tiler1.sub_strides_el() == std::vector<uint64_t>{1});
  CHECK(tiler1.tile_strides_el() == std::vector<uint64_t>{1});
  CHECK(tiler1.sub_tile_coord_offsets() == std::vector<uint64_t>{1});

  // Create new subarray
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2[] = {6, 9};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2}, sizeof(sub2), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test correctness of initialization
  CHECK(tiler2.tile_num() == 1);
  CHECK(tiler2.first_sub_tile_coords() == std::vector<uint64_t>{1});
  CHECK(tiler2.sub_strides_el() == std::vector<uint64_t>{1});
  CHECK(tiler2.tile_strides_el() == std::vector<uint64_t>{1});
  CHECK(tiler2.sub_tile_coord_offsets() == std::vector<uint64_t>{1});

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test copy plan, 1D",
    "[DenseTiler][copy_plan][1d]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {1, 10};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {3, 6};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of copy plan for tile 0
  auto copy_plan1_0 = tiler1.copy_plan(0);
  CHECK(copy_plan1_0.copy_el_ == 3);
  CHECK(
      copy_plan1_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_0.sub_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan1_0.tile_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan1_0.sub_start_el_ == 0);
  CHECK(copy_plan1_0.tile_start_el_ == 2);
  CHECK(copy_plan1_0.first_d_ == 0);

  // Test correctness of copy plan for tile 1
  auto copy_plan1_1 = tiler1.copy_plan(1);
  CHECK(copy_plan1_1.copy_el_ == 1);
  CHECK(
      copy_plan1_1.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_1.sub_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan1_1.tile_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan1_1.sub_start_el_ == 3);
  CHECK(copy_plan1_1.tile_start_el_ == 0);
  CHECK(copy_plan1_1.first_d_ == 0);

  // Create new subarray
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2[] = {7, 8};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2}, sizeof(sub2), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test correctness of copy plan for tile 0
  auto copy_plan2 = tiler2.copy_plan(0);
  CHECK(copy_plan2.copy_el_ == 2);
  CHECK(copy_plan2.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan2.sub_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan2.tile_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan2.sub_start_el_ == 0);
  CHECK(copy_plan2.tile_start_el_ == 1);
  CHECK(copy_plan2.first_d_ == 0);

  // Create new subarray (col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3[] = {7, 8};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3}, sizeof(sub3), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test correctness of copy plan for tile 0
  auto copy_plan3 = tiler3.copy_plan(0);
  CHECK(copy_plan3.copy_el_ == 2);
  CHECK(copy_plan3.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan3.sub_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan3.tile_strides_el_ == std::vector<uint64_t>{1});
  CHECK(copy_plan3.sub_start_el_ == 0);
  CHECK(copy_plan3.tile_start_el_ == 1);
  CHECK(copy_plan1_0.first_d_ == 0);

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, 1D",
    "[DenseTiler][get_tile][1d]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {1, 10};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {3, 6};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(!tiler1.get_tile(0, "foo", &tile1_0).ok());
  CHECK(!tiler1.get_tile(10, "a", &tile1_0).ok());
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0 = {fill_value, fill_value, 1, 2, 3};
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1 = {
      4, fill_value, fill_value, fill_value, fill_value};
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Create new subarray
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2[] = {7, 10};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2}, sizeof(sub2), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test get tile 0
  Tile tile2;
  CHECK(tiler2.get_tile(0, "a", &tile2).ok());
  std::vector<int32_t> c_data2 = {fill_value, 1, 2, 3, 4};
  CHECK(check_tile<int32_t>(&tile2, c_data2));

  // Create new subarray (col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3[] = {7, 10};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3}, sizeof(sub3), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test get tile 0
  Tile tile3;
  CHECK(tiler3.get_tile(0, "a", &tile3).ok());
  std::vector<int32_t> c_data3 = {fill_value, 1, 2, 3, 4};
  CHECK(check_tile<int32_t>(&tile3, c_data3));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, 1D, tile exceeding array domain",
    "[DenseTiler][get_tile][1d][exceeding_domain]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {1, 8};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {3, 6};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(!tiler1.get_tile(0, "foo", &tile1_0).ok());
  CHECK(!tiler1.get_tile(10, "a", &tile1_0).ok());
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0 = {fill_value, fill_value, 1, 2, 3};
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1 = {
      4, fill_value, fill_value, fill_value, fill_value};
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, 1D, negative domain",
    "[DenseTiler][get_tile][1d][exceeding_domain]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {-4, 5};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {-2, 1};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(!tiler1.get_tile(0, "foo", &tile1_0).ok());
  CHECK(!tiler1.get_tile(10, "a", &tile1_0).ok());
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0 = {fill_value, fill_value, 1, 2, 3};
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1 = {
      4, fill_value, fill_value, fill_value, fill_value};
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test initialization, 2D, (row, row)",
    "[DenseTiler][init][2d][row-row]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 6};
  int32_t sub1_1[] = {18, 22};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of initialization
  CHECK(tiler1.tile_num() == 4);
  CHECK(tiler1.first_sub_tile_coords() == std::vector<uint64_t>{0, 1});
  CHECK(tiler1.sub_strides_el() == std::vector<uint64_t>{5, 1});
  CHECK(tiler1.tile_strides_el() == std::vector<uint64_t>{10, 1});
  CHECK(tiler1.sub_tile_coord_offsets() == std::vector<uint64_t>{2, 1});

  // Create subarray (single tile)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2_0[] = {7, 9};
  int32_t sub2_1[] = {23, 27};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2_0, sub2_1}, sizeof(sub2_0), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test correctness of initialization
  CHECK(tiler2.tile_num() == 1);
  CHECK(tiler2.first_sub_tile_coords() == std::vector<uint64_t>{1, 2});
  CHECK(tiler2.sub_strides_el() == std::vector<uint64_t>{5, 1});
  CHECK(tiler2.tile_strides_el() == std::vector<uint64_t>{10, 1});
  CHECK(tiler2.sub_tile_coord_offsets() == std::vector<uint64_t>{1, 1});

  // Create subarray (multiple tiles, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3_0[] = {4, 6};
  int32_t sub3_1[] = {18, 22};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3_0, sub3_1}, sizeof(sub3_0), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test correctness of initialization
  CHECK(tiler3.tile_num() == 4);
  CHECK(tiler3.first_sub_tile_coords() == std::vector<uint64_t>{0, 1});
  CHECK(tiler3.sub_strides_el() == std::vector<uint64_t>{1, 3});
  CHECK(tiler3.tile_strides_el() == std::vector<uint64_t>{10, 1});
  CHECK(tiler3.sub_tile_coord_offsets() == std::vector<uint64_t>{2, 1});

  // Create subarray (single tile, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub4_0[] = {7, 10};
  int32_t sub4_1[] = {23, 27};
  Subarray subarray4(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub4_0, sub4_1}, sizeof(sub4_0), &subarray4);

  // Create DenseTiler
  DenseTiler<int32_t> tiler4(buffers, &subarray4);

  // Test correctness of initialization
  CHECK(tiler4.tile_num() == 1);
  CHECK(tiler4.first_sub_tile_coords() == std::vector<uint64_t>{1, 2});
  CHECK(tiler4.sub_strides_el() == std::vector<uint64_t>{1, 4});
  CHECK(tiler4.tile_strides_el() == std::vector<uint64_t>{10, 1});
  CHECK(tiler4.sub_tile_coord_offsets() == std::vector<uint64_t>{1, 1});

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test initialization, 2D, (col, col)",
    "[DenseTiler][init][2d][col-col]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_COL_MAJOR,
      TILEDB_COL_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 6};
  int32_t sub1_1[] = {18, 22};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of initialization
  CHECK(tiler1.tile_num() == 4);
  CHECK(tiler1.first_sub_tile_coords() == std::vector<uint64_t>{0, 1});
  CHECK(tiler1.sub_strides_el() == std::vector<uint64_t>{5, 1});
  CHECK(tiler1.tile_strides_el() == std::vector<uint64_t>{1, 5});
  CHECK(tiler1.sub_tile_coord_offsets() == std::vector<uint64_t>{1, 2});

  // Create subarray (single tile)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2_0[] = {7, 9};
  int32_t sub2_1[] = {23, 27};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2_0, sub2_1}, sizeof(sub2_0), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test correctness of initialization
  CHECK(tiler2.tile_num() == 1);
  CHECK(tiler2.first_sub_tile_coords() == std::vector<uint64_t>{1, 2});
  CHECK(tiler2.sub_strides_el() == std::vector<uint64_t>{5, 1});
  CHECK(tiler2.tile_strides_el() == std::vector<uint64_t>{1, 5});
  CHECK(tiler2.sub_tile_coord_offsets() == std::vector<uint64_t>{1, 1});

  // Create subarray (multiple tiles, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3_0[] = {4, 6};
  int32_t sub3_1[] = {18, 22};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3_0, sub3_1}, sizeof(sub3_0), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test correctness of initialization
  CHECK(tiler3.tile_num() == 4);
  CHECK(tiler3.first_sub_tile_coords() == std::vector<uint64_t>{0, 1});
  CHECK(tiler3.sub_strides_el() == std::vector<uint64_t>{1, 3});
  CHECK(tiler3.tile_strides_el() == std::vector<uint64_t>{1, 5});
  CHECK(tiler3.sub_tile_coord_offsets() == std::vector<uint64_t>{1, 2});

  // Create subarray (single tile, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub4_0[] = {7, 10};
  int32_t sub4_1[] = {23, 27};
  Subarray subarray4(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub4_0, sub4_1}, sizeof(sub4_0), &subarray4);

  // Create DenseTiler
  DenseTiler<int32_t> tiler4(buffers, &subarray4);

  // Test correctness of initialization
  CHECK(tiler4.tile_num() == 1);
  CHECK(tiler4.first_sub_tile_coords() == std::vector<uint64_t>{1, 2});
  CHECK(tiler4.sub_strides_el() == std::vector<uint64_t>{1, 4});
  CHECK(tiler4.tile_strides_el() == std::vector<uint64_t>{1, 5});
  CHECK(tiler4.sub_tile_coord_offsets() == std::vector<uint64_t>{1, 1});

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test copy plan, 2D, (row, row)",
    "[DenseTiler][copy_plan][2d][row-row]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 6};
  int32_t sub1_1[] = {18, 22};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of copy plan for tile 0
  auto copy_plan1_0 = tiler1.copy_plan(0);
  CHECK(copy_plan1_0.copy_el_ == 3);
  CHECK(
      copy_plan1_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 1}});
  CHECK(copy_plan1_0.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_0.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_0.sub_start_el_ == 0);
  CHECK(copy_plan1_0.tile_start_el_ == 37);
  CHECK(copy_plan1_0.first_d_ == 0);

  // Test correctness of copy plan for tile 1
  auto copy_plan1_1 = tiler1.copy_plan(1);
  CHECK(copy_plan1_1.copy_el_ == 2);
  CHECK(
      copy_plan1_1.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 1}});
  CHECK(copy_plan1_1.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_1.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_1.sub_start_el_ == 3);
  CHECK(copy_plan1_1.tile_start_el_ == 30);
  CHECK(copy_plan1_1.first_d_ == 0);

  // Test correctness of copy plan for tile 2
  auto copy_plan1_2 = tiler1.copy_plan(2);
  CHECK(copy_plan1_2.copy_el_ == 3);
  CHECK(
      copy_plan1_2.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_2.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_2.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_2.sub_start_el_ == 10);
  CHECK(copy_plan1_2.tile_start_el_ == 7);
  CHECK(copy_plan1_2.first_d_ == 0);

  // Test correctness of copy plan for tile 3
  auto copy_plan1_3 = tiler1.copy_plan(3);
  CHECK(copy_plan1_3.copy_el_ == 2);
  CHECK(
      copy_plan1_3.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_3.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_3.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_3.sub_start_el_ == 13);
  CHECK(copy_plan1_3.tile_start_el_ == 0);
  CHECK(copy_plan1_3.first_d_ == 0);

  // Create subarray (single tile)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2_0[] = {3, 5};
  int32_t sub2_1[] = {13, 18};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2_0, sub2_1}, sizeof(sub2_0), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test correctness of copy plan for tile 0
  auto copy_plan2_0 = tiler2.copy_plan(0);
  CHECK(copy_plan2_0.copy_el_ == 6);
  CHECK(
      copy_plan2_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 2}});
  CHECK(copy_plan2_0.sub_strides_el_ == std::vector<uint64_t>{6, 1});
  CHECK(copy_plan2_0.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan2_0.sub_start_el_ == 0);
  CHECK(copy_plan2_0.tile_start_el_ == 22);
  CHECK(copy_plan2_0.first_d_ == 0);

  // Create subarray (multiple tiles, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3_0[] = {4, 6};
  int32_t sub3_1[] = {18, 22};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3_0, sub3_1}, sizeof(sub3_0), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test correctness of copy plan for tile 0
  auto copy_plan3_0 = tiler3.copy_plan(0);
  CHECK(copy_plan3_0.copy_el_ == 1);
  CHECK(
      copy_plan3_0.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 1}, {0, 2}});
  CHECK(copy_plan3_0.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_0.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan3_0.sub_start_el_ == 0);
  CHECK(copy_plan3_0.tile_start_el_ == 37);
  CHECK(copy_plan3_0.first_d_ == 0);

  // Test correctness of copy plan for tile 1
  auto copy_plan3_1 = tiler3.copy_plan(1);
  CHECK(copy_plan3_1.copy_el_ == 1);
  CHECK(
      copy_plan3_1.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 1}, {0, 1}});
  CHECK(copy_plan3_1.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_1.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan3_1.sub_start_el_ == 9);
  CHECK(copy_plan3_1.tile_start_el_ == 30);
  CHECK(copy_plan3_1.first_d_ == 0);

  // Test correctness of copy plan for tile 2
  auto copy_plan3_2 = tiler3.copy_plan(2);
  CHECK(copy_plan3_2.copy_el_ == 1);
  CHECK(
      copy_plan3_2.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 0}, {0, 2}});
  CHECK(copy_plan3_2.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_2.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan3_2.sub_start_el_ == 2);
  CHECK(copy_plan3_2.tile_start_el_ == 7);
  CHECK(copy_plan3_2.first_d_ == 0);

  // Test correctness of copy plan for tile 3
  auto copy_plan3_3 = tiler3.copy_plan(3);
  CHECK(copy_plan3_3.copy_el_ == 1);
  CHECK(
      copy_plan3_3.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 0}, {0, 1}});
  CHECK(copy_plan3_3.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_3.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan3_3.sub_start_el_ == 11);
  CHECK(copy_plan3_3.tile_start_el_ == 0);
  CHECK(copy_plan3_3.first_d_ == 0);

  // Create subarray (single tile, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub4_0[] = {3, 5};
  int32_t sub4_1[] = {13, 18};
  Subarray subarray4(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub4_0, sub4_1}, sizeof(sub4_0), &subarray4);

  // Create DenseTiler
  DenseTiler<int32_t> tiler4(buffers, &subarray4);

  // Test correctness of copy plan for tile 0
  auto copy_plan4_0 = tiler4.copy_plan(0);
  CHECK(copy_plan4_0.copy_el_ == 1);
  CHECK(
      copy_plan4_0.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 2}, {0, 5}});
  CHECK(copy_plan4_0.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan4_0.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan4_0.sub_start_el_ == 0);
  CHECK(copy_plan4_0.tile_start_el_ == 22);
  CHECK(copy_plan4_0.first_d_ == 0);

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test copy plan, 2D, (col, col)",
    "[DenseTiler][copy_plan][2d][col-col]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_COL_MAJOR,
      TILEDB_COL_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 6};
  int32_t sub1_1[] = {18, 22};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of copy plan for tile 0
  auto copy_plan1_0 = tiler1.copy_plan(0);
  CHECK(copy_plan1_0.copy_el_ == 1);
  CHECK(
      copy_plan1_0.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 1}, {0, 2}});
  CHECK(copy_plan1_0.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_0.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_0.sub_start_el_ == 0);
  CHECK(copy_plan1_0.tile_start_el_ == 38);
  CHECK(copy_plan1_0.first_d_ == 0);

  // Test correctness of copy plan for tile 1
  auto copy_plan1_1 = tiler1.copy_plan(1);
  CHECK(copy_plan1_1.copy_el_ == 1);
  CHECK(
      copy_plan1_1.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 0}, {0, 2}});
  CHECK(copy_plan1_1.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_1.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_1.sub_start_el_ == 10);
  CHECK(copy_plan1_1.tile_start_el_ == 35);
  CHECK(copy_plan1_1.first_d_ == 0);

  // Test correctness of copy plan for tile 2
  auto copy_plan1_2 = tiler1.copy_plan(2);
  CHECK(copy_plan1_2.copy_el_ == 1);
  CHECK(
      copy_plan1_2.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 1}, {0, 1}});
  CHECK(copy_plan1_2.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_2.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_2.sub_start_el_ == 3);
  CHECK(copy_plan1_2.tile_start_el_ == 3);
  CHECK(copy_plan1_2.first_d_ == 0);

  // Test correctness of copy plan for tile 3
  auto copy_plan1_3 = tiler1.copy_plan(3);
  CHECK(copy_plan1_3.copy_el_ == 1);
  CHECK(
      copy_plan1_3.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 0}, {0, 1}});
  CHECK(copy_plan1_3.sub_strides_el_ == std::vector<uint64_t>{5, 1});
  CHECK(copy_plan1_3.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_3.sub_start_el_ == 13);
  CHECK(copy_plan1_3.tile_start_el_ == 0);
  CHECK(copy_plan1_3.first_d_ == 0);

  // Create subarray (single tile)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2_0[] = {3, 5};
  int32_t sub2_1[] = {13, 18};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2_0, sub2_1}, sizeof(sub2_0), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test correctness of copy plan for tile 0
  auto copy_plan2_0 = tiler2.copy_plan(0);
  CHECK(copy_plan2_0.copy_el_ == 1);
  CHECK(
      copy_plan2_0.dim_ranges_ ==
      std::vector<std::array<uint64_t, 2>>{{0, 2}, {0, 5}});
  CHECK(copy_plan2_0.sub_strides_el_ == std::vector<uint64_t>{6, 1});
  CHECK(copy_plan2_0.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan2_0.sub_start_el_ == 0);
  CHECK(copy_plan2_0.tile_start_el_ == 12);
  CHECK(copy_plan2_0.first_d_ == 0);

  // Create subarray (multiple tiles, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3_0[] = {4, 6};
  int32_t sub3_1[] = {18, 22};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3_0, sub3_1}, sizeof(sub3_0), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test correctness of copy plan for tile 0
  auto copy_plan3_0 = tiler3.copy_plan(0);
  CHECK(copy_plan3_0.copy_el_ == 2);
  CHECK(
      copy_plan3_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 2}});
  CHECK(copy_plan3_0.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_0.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan3_0.sub_start_el_ == 0);
  CHECK(copy_plan3_0.tile_start_el_ == 38);
  CHECK(copy_plan3_0.first_d_ == 1);

  // Test correctness of copy plan for tile 1
  auto copy_plan3_1 = tiler3.copy_plan(1);
  CHECK(copy_plan3_1.copy_el_ == 1);
  CHECK(
      copy_plan3_1.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 2}});
  CHECK(copy_plan3_1.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_1.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan3_1.sub_start_el_ == 2);
  CHECK(copy_plan3_1.tile_start_el_ == 35);
  CHECK(copy_plan3_1.first_d_ == 1);

  // Test correctness of copy plan for tile 2
  auto copy_plan3_2 = tiler3.copy_plan(2);
  CHECK(copy_plan3_2.copy_el_ == 2);
  CHECK(
      copy_plan3_2.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 1}});
  CHECK(copy_plan3_2.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_2.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan3_2.sub_start_el_ == 9);
  CHECK(copy_plan3_2.tile_start_el_ == 3);
  CHECK(copy_plan3_2.first_d_ == 1);

  // Test correctness of copy plan for tile 3
  auto copy_plan3_3 = tiler3.copy_plan(3);
  CHECK(copy_plan3_3.copy_el_ == 1);
  CHECK(
      copy_plan3_3.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 1}});
  CHECK(copy_plan3_3.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan3_3.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan3_3.sub_start_el_ == 11);
  CHECK(copy_plan3_3.tile_start_el_ == 0);
  CHECK(copy_plan3_3.first_d_ == 1);

  // Create subarray (single tile, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub4_0[] = {3, 5};
  int32_t sub4_1[] = {13, 18};
  Subarray subarray4(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub4_0, sub4_1}, sizeof(sub4_0), &subarray4);

  // Create DenseTiler
  DenseTiler<int32_t> tiler4(buffers, &subarray4);

  // Test correctness of copy plan for tile 0
  auto copy_plan4_0 = tiler4.copy_plan(0);
  CHECK(copy_plan4_0.copy_el_ == 3);
  CHECK(
      copy_plan4_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 5}});
  CHECK(copy_plan4_0.sub_strides_el_ == std::vector<uint64_t>{1, 3});
  CHECK(copy_plan4_0.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan4_0.sub_start_el_ == 0);
  CHECK(copy_plan4_0.tile_start_el_ == 12);
  CHECK(copy_plan4_0.first_d_ == 1);

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test copy plan optimization, 2D, (row, row)",
    "[DenseTiler][copy_plan][optimization][2d][row-row]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 9};
  int32_t sub1_1[] = {11, 20};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of copy plan for tile 0
  auto copy_plan1_0 = tiler1.copy_plan(0);
  CHECK(copy_plan1_0.copy_el_ == 20);
  CHECK(
      copy_plan1_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_0.sub_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_0.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_0.sub_start_el_ == 0);
  CHECK(copy_plan1_0.tile_start_el_ == 30);
  CHECK(copy_plan1_0.first_d_ == 0);

  // Test correctness of copy plan for tile 1
  auto copy_plan1_1 = tiler1.copy_plan(1);
  CHECK(copy_plan1_1.copy_el_ == 40);
  CHECK(
      copy_plan1_1.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_1.sub_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_1.tile_strides_el_ == std::vector<uint64_t>{10, 1});
  CHECK(copy_plan1_1.sub_start_el_ == 20);
  CHECK(copy_plan1_1.tile_start_el_ == 0);
  CHECK(copy_plan1_1.first_d_ == 0);

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test copy plan optimization, 2D, (col, col)",
    "[DenseTiler][copy_plan][optimization][2d][col-col]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_COL_MAJOR,
      TILEDB_COL_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 2, 3, 4};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {1, 5};
  int32_t sub1_1[] = {8, 12};
  Subarray subarray1(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test correctness of copy plan for tile 0
  auto copy_plan1_0 = tiler1.copy_plan(0);
  CHECK(copy_plan1_0.copy_el_ == 15);
  CHECK(
      copy_plan1_0.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_0.sub_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_0.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_0.sub_start_el_ == 0);
  CHECK(copy_plan1_0.tile_start_el_ == 35);
  CHECK(copy_plan1_0.first_d_ == 1);

  // Test correctness of copy plan for tile 0
  auto copy_plan1_1 = tiler1.copy_plan(1);
  CHECK(copy_plan1_1.copy_el_ == 10);
  CHECK(
      copy_plan1_1.dim_ranges_ == std::vector<std::array<uint64_t, 2>>{{0, 0}});
  CHECK(copy_plan1_1.sub_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_1.tile_strides_el_ == std::vector<uint64_t>{1, 5});
  CHECK(copy_plan1_1.sub_start_el_ == 15);
  CHECK(copy_plan1_1.tile_start_el_ == 0);
  CHECK(copy_plan1_1.first_d_ == 1);

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, 2D, (row, row)",
    "[DenseTiler][get_tile][2d][row-row]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 6};
  int32_t sub1_1[] = {18, 22};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0(50);
  for (int i = 0; i <= 36; ++i)
    c_data1_0[i] = fill_value;
  for (int i = 37; i <= 39; ++i)
    c_data1_0[i] = i - 36;
  for (int i = 40; i <= 46; ++i)
    c_data1_0[i] = fill_value;
  for (int i = 47; i <= 49; ++i)
    c_data1_0[i] = i - 41;
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1(50);
  for (int i = 0; i <= 29; ++i)
    c_data1_1[i] = fill_value;
  for (int i = 30; i <= 31; ++i)
    c_data1_1[i] = i - 26;
  for (int i = 32; i <= 39; ++i)
    c_data1_1[i] = fill_value;
  for (int i = 40; i <= 41; ++i)
    c_data1_1[i] = i - 31;
  for (int i = 42; i <= 49; ++i)
    c_data1_1[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Test get tile 2
  Tile tile1_2;
  CHECK(tiler1.get_tile(2, "a", &tile1_2).ok());
  std::vector<int32_t> c_data1_2(50);
  for (int i = 0; i <= 6; ++i)
    c_data1_2[i] = fill_value;
  for (int i = 7; i <= 9; ++i)
    c_data1_2[i] = i + 4;
  for (int i = 10; i <= 49; ++i)
    c_data1_2[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_2, c_data1_2));

  // Test get tile 3
  Tile tile1_3;
  CHECK(tiler1.get_tile(3, "a", &tile1_3).ok());
  std::vector<int32_t> c_data1_3(50);
  for (int i = 0; i <= 1; ++i)
    c_data1_3[i] = i + 14;
  for (int i = 2; i <= 49; ++i)
    c_data1_3[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_3, c_data1_3));

  // Create subarray (single tile)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2_0[] = {3, 5};
  int32_t sub2_1[] = {13, 18};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2_0, sub2_1}, sizeof(sub2_0), &subarray2);

  // Create DenseTiler
  buff_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test get tile 0
  Tile tile2_0;
  CHECK(tiler2.get_tile(0, "a", &tile2_0).ok());
  std::vector<int32_t> c_data2_0(50);
  for (int i = 0; i <= 21; ++i)
    c_data2_0[i] = fill_value;
  for (int i = 22; i <= 27; ++i)
    c_data2_0[i] = i - 21;
  for (int i = 28; i <= 31; ++i)
    c_data2_0[i] = fill_value;
  for (int i = 32; i <= 37; ++i)
    c_data2_0[i] = i - 25;
  for (int i = 38; i <= 41; ++i)
    c_data2_0[i] = fill_value;
  for (int i = 42; i <= 47; ++i)
    c_data2_0[i] = i - 29;
  for (int i = 48; i <= 49; ++i)
    c_data2_0[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile2_0, c_data2_0));

  // Create subarray (multiple tiles, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3_0[] = {4, 6};
  int32_t sub3_1[] = {18, 22};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3_0, sub3_1}, sizeof(sub3_0), &subarray3);

  // Create DenseTiler
  buff_a = {1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15};
  buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test get tile 0
  Tile tile3_0;
  CHECK(tiler3.get_tile(0, "a", &tile3_0).ok());
  std::vector<int32_t> c_data3_0(50);
  for (int i = 0; i <= 36; ++i)
    c_data3_0[i] = fill_value;
  for (int i = 37; i <= 39; ++i)
    c_data3_0[i] = i - 36;
  for (int i = 40; i <= 46; ++i)
    c_data3_0[i] = fill_value;
  for (int i = 47; i <= 49; ++i)
    c_data3_0[i] = i - 41;
  CHECK(check_tile<int32_t>(&tile3_0, c_data3_0));

  // Test get tile 1
  Tile tile3_1;
  CHECK(tiler3.get_tile(1, "a", &tile3_1).ok());
  std::vector<int32_t> c_data3_1(50);
  for (int i = 0; i <= 29; ++i)
    c_data3_1[i] = fill_value;
  for (int i = 30; i <= 31; ++i)
    c_data3_1[i] = i - 26;
  for (int i = 32; i <= 39; ++i)
    c_data3_1[i] = fill_value;
  for (int i = 40; i <= 41; ++i)
    c_data3_1[i] = i - 31;
  for (int i = 42; i <= 49; ++i)
    c_data3_1[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile3_1, c_data3_1));

  // Test get tile 2
  Tile tile3_2;
  CHECK(tiler3.get_tile(2, "a", &tile3_2).ok());
  std::vector<int32_t> c_data3_2(50);
  for (int i = 0; i <= 6; ++i)
    c_data3_2[i] = fill_value;
  for (int i = 7; i <= 9; ++i)
    c_data3_2[i] = i + 4;
  for (int i = 10; i <= 49; ++i)
    c_data3_2[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile3_2, c_data3_2));

  // Test get tile 3
  Tile tile3_3;
  CHECK(tiler3.get_tile(3, "a", &tile3_3).ok());
  std::vector<int32_t> c_data3_3(50);
  for (int i = 0; i <= 1; ++i)
    c_data3_3[i] = i + 14;
  for (int i = 2; i <= 49; ++i)
    c_data3_3[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile3_3, c_data3_3));

  // Create subarray (single tile, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub4_0[] = {3, 5};
  int32_t sub4_1[] = {13, 18};
  Subarray subarray4(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub4_0, sub4_1}, sizeof(sub4_0), &subarray4);

  // Create DenseTiler
  buff_a = {1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17, 6, 12, 18};
  buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);
  DenseTiler<int32_t> tiler4(buffers, &subarray4);

  // Test get tile 0
  Tile tile4_0;
  CHECK(tiler4.get_tile(0, "a", &tile4_0).ok());
  std::vector<int32_t> c_data4_0(50);
  for (int i = 0; i <= 21; ++i)
    c_data4_0[i] = fill_value;
  for (int i = 22; i <= 27; ++i)
    c_data4_0[i] = i - 21;
  for (int i = 28; i <= 31; ++i)
    c_data4_0[i] = fill_value;
  for (int i = 32; i <= 37; ++i)
    c_data4_0[i] = i - 25;
  for (int i = 38; i <= 41; ++i)
    c_data4_0[i] = fill_value;
  for (int i = 42; i <= 47; ++i)
    c_data4_0[i] = i - 29;
  for (int i = 48; i <= 49; ++i)
    c_data4_0[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile4_0, c_data4_0));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, 2D, (col, col)",
    "[DenseTiler][get_tile][2d][col-col]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_COL_MAJOR,
      TILEDB_COL_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 6};
  int32_t sub1_1[] = {18, 22};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0(50);
  for (int i = 0; i <= 37; ++i)
    c_data1_0[i] = fill_value;
  c_data1_0[38] = 1;
  c_data1_0[39] = 6;
  for (int i = 40; i <= 42; ++i)
    c_data1_0[i] = fill_value;
  c_data1_0[43] = 2;
  c_data1_0[44] = 7;
  for (int i = 45; i <= 47; ++i)
    c_data1_0[i] = fill_value;
  c_data1_0[48] = 3;
  c_data1_0[49] = 8;
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1(50);
  for (int i = 0; i <= 34; ++i)
    c_data1_1[i] = fill_value;
  c_data1_1[35] = 11;
  for (int i = 36; i <= 39; ++i)
    c_data1_1[i] = fill_value;
  c_data1_1[40] = 12;
  for (int i = 41; i <= 44; ++i)
    c_data1_1[i] = fill_value;
  c_data1_1[45] = 13;
  for (int i = 46; i <= 49; ++i)
    c_data1_1[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Test get tile 2
  Tile tile1_2;
  CHECK(tiler1.get_tile(2, "a", &tile1_2).ok());
  std::vector<int32_t> c_data1_2(50);
  for (int i = 0; i <= 2; ++i)
    c_data1_2[i] = fill_value;
  c_data1_2[3] = 4;
  c_data1_2[4] = 9;
  for (int i = 5; i <= 7; ++i)
    c_data1_2[i] = fill_value;
  c_data1_2[8] = 5;
  c_data1_2[9] = 10;
  for (int i = 10; i <= 49; ++i)
    c_data1_2[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_2, c_data1_2));

  // Test get tile 3
  Tile tile1_3;
  CHECK(tiler1.get_tile(3, "a", &tile1_3).ok());
  std::vector<int32_t> c_data1_3(50);
  c_data1_3[0] = 14;
  for (int i = 1; i <= 4; ++i)
    c_data1_3[i] = fill_value;
  c_data1_3[5] = 15;
  for (int i = 6; i <= 49; ++i)
    c_data1_3[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_3, c_data1_3));

  // Create subarray (single tile)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2_0[] = {3, 5};
  int32_t sub2_1[] = {13, 18};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2_0, sub2_1}, sizeof(sub2_0), &subarray2);

  // Create DenseTiler
  buff_a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test get tile 0
  Tile tile2_0;
  CHECK(tiler2.get_tile(0, "a", &tile2_0).ok());
  std::vector<int32_t> c_data2_0(50);
  for (int i = 0; i <= 11; ++i)
    c_data2_0[i] = fill_value;
  c_data2_0[12] = 1;
  c_data2_0[13] = 7;
  c_data2_0[14] = 13;
  for (int i = 15; i <= 16; ++i)
    c_data2_0[i] = fill_value;
  c_data2_0[17] = 2;
  c_data2_0[18] = 8;
  c_data2_0[19] = 14;
  for (int i = 20; i <= 21; ++i)
    c_data2_0[i] = fill_value;
  c_data2_0[22] = 3;
  c_data2_0[23] = 9;
  c_data2_0[24] = 15;
  for (int i = 25; i <= 26; ++i)
    c_data2_0[i] = fill_value;
  c_data2_0[27] = 4;
  c_data2_0[28] = 10;
  c_data2_0[29] = 16;
  for (int i = 30; i <= 31; ++i)
    c_data2_0[i] = fill_value;
  c_data2_0[32] = 5;
  c_data2_0[33] = 11;
  c_data2_0[34] = 17;
  for (int i = 35; i <= 36; ++i)
    c_data2_0[i] = fill_value;
  c_data2_0[37] = 6;
  c_data2_0[38] = 12;
  c_data2_0[39] = 18;
  for (int i = 40; i <= 49; ++i)
    c_data2_0[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile2_0, c_data2_0));

  // Create subarray (multiple tiles, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3_0[] = {4, 6};
  int32_t sub3_1[] = {18, 22};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3_0, sub3_1}, sizeof(sub3_0), &subarray3);

  // Create DenseTiler
  buff_a = {1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15};
  buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test get tile 0
  Tile tile3_0;
  CHECK(tiler3.get_tile(0, "a", &tile3_0).ok());
  std::vector<int32_t> c_data3_0(50);
  for (int i = 0; i <= 37; ++i)
    c_data3_0[i] = fill_value;
  c_data3_0[38] = 1;
  c_data3_0[39] = 6;
  for (int i = 40; i <= 42; ++i)
    c_data3_0[i] = fill_value;
  c_data3_0[43] = 2;
  c_data3_0[44] = 7;
  for (int i = 45; i <= 47; ++i)
    c_data3_0[i] = fill_value;
  c_data3_0[48] = 3;
  c_data3_0[49] = 8;
  CHECK(check_tile<int32_t>(&tile3_0, c_data3_0));

  // Test get tile 1
  Tile tile3_1;
  CHECK(tiler3.get_tile(1, "a", &tile3_1).ok());
  std::vector<int32_t> c_data3_1(50);
  for (int i = 0; i <= 34; ++i)
    c_data3_1[i] = fill_value;
  c_data3_1[35] = 11;
  for (int i = 36; i <= 39; ++i)
    c_data3_1[i] = fill_value;
  c_data3_1[40] = 12;
  for (int i = 41; i <= 44; ++i)
    c_data3_1[i] = fill_value;
  c_data3_1[45] = 13;
  for (int i = 46; i <= 49; ++i)
    c_data3_1[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile3_1, c_data3_1));

  // Test get tile 2
  Tile tile3_2;
  CHECK(tiler3.get_tile(2, "a", &tile3_2).ok());
  std::vector<int32_t> c_data3_2(50);
  for (int i = 0; i <= 2; ++i)
    c_data3_2[i] = fill_value;
  c_data3_2[3] = 4;
  c_data3_2[4] = 9;
  for (int i = 5; i <= 7; ++i)
    c_data3_2[i] = fill_value;
  c_data3_2[8] = 5;
  c_data3_2[9] = 10;
  for (int i = 10; i <= 49; ++i)
    c_data3_2[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile3_2, c_data3_2));

  // Test get tile 3
  Tile tile3_3;
  CHECK(tiler3.get_tile(3, "a", &tile3_3).ok());
  std::vector<int32_t> c_data3_3(50);
  c_data3_3[0] = 14;
  for (int i = 1; i <= 4; ++i)
    c_data3_3[i] = fill_value;
  c_data3_3[5] = 15;
  for (int i = 6; i <= 49; ++i)
    c_data3_3[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile3_3, c_data3_3));

  // Create subarray (single tile, col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub4_0[] = {3, 5};
  int32_t sub4_1[] = {13, 18};
  Subarray subarray4(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub4_0, sub4_1}, sizeof(sub4_0), &subarray4);

  // Create DenseTiler
  buff_a = {1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 5, 11, 17, 6, 12, 18};
  buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);
  DenseTiler<int32_t> tiler4(buffers, &subarray4);

  // Test get tile 0
  Tile tile4_0;
  CHECK(tiler4.get_tile(0, "a", &tile4_0).ok());
  std::vector<int32_t> c_data4_0(50);
  for (int i = 0; i <= 11; ++i)
    c_data4_0[i] = fill_value;
  c_data4_0[12] = 1;
  c_data4_0[13] = 7;
  c_data4_0[14] = 13;
  for (int i = 15; i <= 16; ++i)
    c_data4_0[i] = fill_value;
  c_data4_0[17] = 2;
  c_data4_0[18] = 8;
  c_data4_0[19] = 14;
  for (int i = 20; i <= 21; ++i)
    c_data4_0[i] = fill_value;
  c_data4_0[22] = 3;
  c_data4_0[23] = 9;
  c_data4_0[24] = 15;
  for (int i = 25; i <= 26; ++i)
    c_data4_0[i] = fill_value;
  c_data4_0[27] = 4;
  c_data4_0[28] = 10;
  c_data4_0[29] = 16;
  for (int i = 30; i <= 31; ++i)
    c_data4_0[i] = fill_value;
  c_data4_0[32] = 5;
  c_data4_0[33] = 11;
  c_data4_0[34] = 17;
  for (int i = 35; i <= 36; ++i)
    c_data4_0[i] = fill_value;
  c_data4_0[37] = 6;
  c_data4_0[38] = 12;
  c_data4_0[39] = 18;
  for (int i = 40; i <= 49; ++i)
    c_data4_0[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile4_0, c_data4_0));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile optimization, 2D, (row, row)",
    "[DenseTiler][get_tile][optimization][2d][row-row]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {
      1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
      31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {4, 9};
  int32_t sub1_1[] = {11, 20};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0(50);
  for (int i = 0; i <= 29; ++i)
    c_data1_0[i] = fill_value;
  for (int i = 30; i <= 49; ++i)
    c_data1_0[i] = i - 29;
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1(50);
  for (int i = 0; i <= 39; ++i)
    c_data1_1[i] = i + 21;
  for (int i = 40; i <= 49; ++i)
    c_data1_1[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile optimization, 2D, (col, col)",
    "[DenseTiler][get_tile][optimization][2d][col-col]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom_1[] = {1, 10};
  int32_t d_ext_1 = 5;
  int32_t d_dom_2[] = {1, 30};
  int32_t d_ext_2 = 10;
  create_array(
      array_name,
      {{"d1", TILEDB_INT32, d_dom_1, &d_ext_1},
       {"d2", TILEDB_INT32, d_dom_2, &d_ext_2}},
      {{"a", TILEDB_INT32, 1}},
      TILEDB_COL_MAJOR,
      TILEDB_COL_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1,  2,  3,  4,  5,  6,  7,  8,  9,
                                 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                 19, 20, 21, 22, 23, 24, 25};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray (multiple tiles)
  open_array(array_name, TILEDB_READ);
  int32_t sub1_0[] = {1, 5};
  int32_t sub1_1[] = {8, 12};
  Subarray subarray1(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub1_0, sub1_1}, sizeof(sub1_0), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0(50);
  for (int i = 0; i <= 34; ++i)
    c_data1_0[i] = fill_value;
  for (int i = 35; i <= 49; ++i)
    c_data1_0[i] = i - 34;
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1(50);
  for (int i = 0; i <= 9; ++i)
    c_data1_1[i] = i + 16;
  for (int i = 10; i <= 49; ++i)
    c_data1_1[i] = fill_value;
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, cell val num = 2, 1D",
    "[DenseTiler][get_tile][cell_val_num_2][1d]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {1, 10};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a", TILEDB_INT32, 2}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a = {1, 11, 2, 22, 3, 33, 4, 44};
  uint64_t buff_a_size = sizeof(buff_a);
  buffers["a"] = QueryBuffer(&buff_a[0], nullptr, &buff_a_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {3, 6};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0;
  CHECK(tiler1.get_tile(0, "a", &tile1_0).ok());
  std::vector<int32_t> c_data1_0 = {
      fill_value, fill_value, fill_value, fill_value, 1, 11, 2, 22, 3, 33};
  CHECK(check_tile<int32_t>(&tile1_0, c_data1_0));

  // Test get tile 1
  Tile tile1_1;
  CHECK(tiler1.get_tile(1, "a", &tile1_1).ok());
  std::vector<int32_t> c_data1_1 = {4,
                                    44,
                                    fill_value,
                                    fill_value,
                                    fill_value,
                                    fill_value,
                                    fill_value,
                                    fill_value,
                                    fill_value,
                                    fill_value};
  CHECK(check_tile<int32_t>(&tile1_1, c_data1_1));

  // Create new subarray
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2[] = {7, 10};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2}, sizeof(sub2), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test get tile 0
  Tile tile2;
  CHECK(tiler2.get_tile(0, "a", &tile2).ok());
  std::vector<int32_t> c_data2 = {
      fill_value, fill_value, 1, 11, 2, 22, 3, 33, 4, 44};
  CHECK(check_tile<int32_t>(&tile2, c_data2));

  // Create new subarray (col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3[] = {7, 10};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3}, sizeof(sub3), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test get tile 0
  Tile tile3;
  CHECK(tiler3.get_tile(0, "a", &tile3).ok());
  std::vector<int32_t> c_data3 = {
      fill_value, fill_value, 1, 11, 2, 22, 3, 33, 4, 44};
  CHECK(check_tile<int32_t>(&tile3, c_data3));

  // Clean up
  close_array();
  remove_array(array_name);
}

TEST_CASE_METHOD(
    DenseTilerFx,
    "DenseTiler: Test get tile, 2 attributes, 1D",
    "[DenseTiler][get_tile][2_attributes][1d]") {
  // Create array
  std::string array_name = "dense_tiler";
  int32_t d_dom[] = {1, 10};
  int32_t d_ext = 5;
  create_array(
      array_name,
      {{"d", TILEDB_INT32, d_dom, &d_ext}},
      {{"a1", TILEDB_INT32, 1}, {"a2", TILEDB_FLOAT64, 1}},
      TILEDB_ROW_MAJOR,
      TILEDB_ROW_MAJOR);

  // Create buffers
  std::unordered_map<std::string, QueryBuffer> buffers;
  std::vector<int32_t> buff_a1 = {1, 2, 3, 4};
  uint64_t buff_a1_size = sizeof(buff_a1);
  buffers["a1"] = QueryBuffer(&buff_a1[0], nullptr, &buff_a1_size, nullptr);
  std::vector<double> buff_a2 = {1.1, 2.2, 3.3, 4.4};
  uint64_t buff_a2_size = sizeof(buff_a2);
  buffers["a2"] = QueryBuffer(&buff_a2[0], nullptr, &buff_a2_size, nullptr);

  // Create subarray
  open_array(array_name, TILEDB_READ);
  int32_t sub1[] = {3, 6};
  Subarray subarray1(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub1}, sizeof(sub1), &subarray1);

  // Create DenseTiler
  DenseTiler<int32_t> tiler1(buffers, &subarray1);

  // Test get tile 0
  Tile tile1_0_a1;
  CHECK(tiler1.get_tile(0, "a1", &tile1_0_a1).ok());
  std::vector<int32_t> c_data1_0_a1 = {fill_value, fill_value, 1, 2, 3};
  CHECK(check_tile<int32_t>(&tile1_0_a1, c_data1_0_a1));
  Tile tile1_0_a2;
  CHECK(tiler1.get_tile(0, "a2", &tile1_0_a2).ok());
  std::vector<double> c_data1_0_a2 = {
      double(fill_value), double(fill_value), 1.1, 2.2, 3.3};
  CHECK(check_tile<double>(&tile1_0_a2, c_data1_0_a2));

  // Test get tile 1
  Tile tile1_1_a1;
  CHECK(tiler1.get_tile(1, "a1", &tile1_1_a1).ok());
  std::vector<int32_t> c_data1_1_a1 = {
      4, fill_value, fill_value, fill_value, fill_value};
  CHECK(check_tile<int32_t>(&tile1_1_a1, c_data1_1_a1));
  Tile tile1_1_a2;
  CHECK(tiler1.get_tile(1, "a2", &tile1_1_a2).ok());
  std::vector<double> c_data1_1_a2 = {4.4,
                                      double(fill_value),
                                      double(fill_value),
                                      double(fill_value),
                                      double(fill_value)};
  CHECK(check_tile<double>(&tile1_1_a2, c_data1_1_a2));

  // Create new subarray
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub2[] = {7, 10};
  Subarray subarray2(array_->array_, Layout::ROW_MAJOR);
  add_ranges({sub2}, sizeof(sub2), &subarray2);

  // Create DenseTiler
  DenseTiler<int32_t> tiler2(buffers, &subarray2);

  // Test get tile 0
  Tile tile2_a1;
  CHECK(tiler2.get_tile(0, "a1", &tile2_a1).ok());
  std::vector<int32_t> c_data2_a1 = {fill_value, 1, 2, 3, 4};
  CHECK(check_tile<int32_t>(&tile2_a1, c_data2_a1));
  Tile tile2_a2;
  CHECK(tiler2.get_tile(0, "a2", &tile2_a2).ok());
  std::vector<double> c_data2_a2 = {double(fill_value), 1.1, 2.2, 3.3, 4.4};
  CHECK(check_tile<double>(&tile2_a2, c_data2_a2));

  // Create new subarray (col-major)
  close_array();
  open_array(array_name, TILEDB_READ);
  int32_t sub3[] = {7, 10};
  Subarray subarray3(array_->array_, Layout::COL_MAJOR);
  add_ranges({sub3}, sizeof(sub3), &subarray3);

  // Create DenseTiler
  DenseTiler<int32_t> tiler3(buffers, &subarray3);

  // Test get tile 0
  Tile tile3_a1;
  CHECK(tiler3.get_tile(0, "a1", &tile3_a1).ok());
  std::vector<int32_t> c_data3_a1 = {fill_value, 1, 2, 3, 4};
  CHECK(check_tile<int32_t>(&tile3_a1, c_data3_a1));
  Tile tile3_a2;
  CHECK(tiler3.get_tile(0, "a2", &tile3_a2).ok());
  std::vector<double> c_data3_a2 = {double(fill_value), 1.1, 2.2, 3.3, 4.4};
  CHECK(check_tile<double>(&tile3_a2, c_data3_a2));

  // Clean up
  close_array();
  remove_array(array_name);
}

// TODO DenseTiler: 3D same as all 2D

// TODO: test var size
// TODO: test nullable

// TODO: add TIME datatype to dense arrays?
// TODO: delete offsets_ in domain (dead code) + WriteCellSlabIter
//       + Domain::get_tile_pos