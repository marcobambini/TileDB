/**
 * @file   unit-capi-consolidation.cc
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2017-2018 TileDB Inc.
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
 * Tests the C API async queries.
 */

#include "catch.hpp"
#include "tiledb/sm/c_api/tiledb.h"

#include <cstring>

/**
 * Tests cases where a read query is incomplete or leads to a buffer
 * overflow.
 */

struct IncompleteFx {
  // Constants
  const char* DENSE_ARRAY_NAME = "test_async_dense";
  const char* SPARSE_ARRAY_NAME = "test_async_sparse";

  // TileDB context
  tiledb_ctx_t* ctx_;

  // Constructors/destructors
  IncompleteFx();
  ~IncompleteFx();

  // Functions
  void create_dense_array();
  void create_sparse_array();
  void write_dense_full();
  void write_sparse_full();
  void check_dense_incomplete();
  void check_dense_until_complete();
  void check_dense_unsplittable_overflow();
  void check_dense_unsplittable_complete();
  void check_dense_reset_buffers();
  void check_sparse_incomplete();
  void check_sparse_until_complete();
  void check_sparse_unsplittable_overflow();
  void check_sparse_unsplittable_complete();
  void remove_dense_array();
  void remove_sparse_array();
  void remove_array(const std::string& array_name);
  bool is_array(const std::string& array_name);
};

IncompleteFx::IncompleteFx() {
  ctx_ = nullptr;
  REQUIRE(tiledb_ctx_create(&ctx_, NULL) == TILEDB_OK);
}

IncompleteFx::~IncompleteFx() {
  tiledb_ctx_free(&ctx_);
}

void IncompleteFx::create_dense_array() {
  // Create dimensions
  uint64_t dim_domain[] = {1, 4, 1, 4};
  uint64_t tile_extents[] = {2, 2};
  tiledb_dimension_t* d1;
  int rc = tiledb_dimension_create(
      ctx_, &d1, "d1", TILEDB_UINT64, &dim_domain[0], &tile_extents[0]);
  CHECK(rc == TILEDB_OK);
  tiledb_dimension_t* d2;
  rc = tiledb_dimension_create(
      ctx_, &d2, "d2", TILEDB_UINT64, &dim_domain[2], &tile_extents[1]);
  CHECK(rc == TILEDB_OK);

  // Create domain
  tiledb_domain_t* domain;
  rc = tiledb_domain_create(ctx_, &domain);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_domain_add_dimension(ctx_, domain, d1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_domain_add_dimension(ctx_, domain, d2);
  CHECK(rc == TILEDB_OK);

  // Create attributes
  tiledb_attribute_t* a1;
  rc = tiledb_attribute_create(ctx_, &a1, "a1", TILEDB_INT32);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_compressor(ctx_, a1, TILEDB_BLOSC_LZ, -1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_cell_val_num(ctx_, a1, 1);
  CHECK(rc == TILEDB_OK);
  tiledb_attribute_t* a2;
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_create(ctx_, &a2, "a2", TILEDB_CHAR);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_compressor(ctx_, a2, TILEDB_GZIP, -1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_cell_val_num(ctx_, a2, TILEDB_VAR_NUM);
  CHECK(rc == TILEDB_OK);
  tiledb_attribute_t* a3;
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_create(ctx_, &a3, "a3", TILEDB_FLOAT32);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_compressor(ctx_, a3, TILEDB_ZSTD, -1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_cell_val_num(ctx_, a3, 2);
  CHECK(rc == TILEDB_OK);

  // Create array schema
  tiledb_array_schema_t* array_schema;
  rc = tiledb_array_schema_create(ctx_, &array_schema, TILEDB_DENSE);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_cell_order(ctx_, array_schema, TILEDB_ROW_MAJOR);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_tile_order(ctx_, array_schema, TILEDB_ROW_MAJOR);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_domain(ctx_, array_schema, domain);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_add_attribute(ctx_, array_schema, a1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_add_attribute(ctx_, array_schema, a2);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_add_attribute(ctx_, array_schema, a3);
  CHECK(rc == TILEDB_OK);

  // Check array schema
  rc = tiledb_array_schema_check(ctx_, array_schema);
  CHECK(rc == TILEDB_OK);

  // Create array
  rc = tiledb_array_create(ctx_, DENSE_ARRAY_NAME, array_schema);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_attribute_free(&a1);
  tiledb_attribute_free(&a2);
  tiledb_attribute_free(&a3);
  tiledb_dimension_free(&d1);
  tiledb_dimension_free(&d2);
  tiledb_domain_free(&domain);
  tiledb_array_schema_free(&array_schema);
}

void IncompleteFx::create_sparse_array() {
  // Create dimensions
  uint64_t dim_domain[] = {1, 4, 1, 4};
  uint64_t tile_extents[] = {2, 2};
  tiledb_dimension_t* d1;
  int rc = tiledb_dimension_create(
      ctx_, &d1, "d1", TILEDB_UINT64, &dim_domain[0], &tile_extents[0]);
  CHECK(rc == TILEDB_OK);
  tiledb_dimension_t* d2;
  rc = tiledb_dimension_create(
      ctx_, &d2, "d2", TILEDB_UINT64, &dim_domain[2], &tile_extents[1]);
  CHECK(rc == TILEDB_OK);

  // Create domain
  tiledb_domain_t* domain;
  rc = tiledb_domain_create(ctx_, &domain);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_domain_add_dimension(ctx_, domain, d1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_domain_add_dimension(ctx_, domain, d2);
  CHECK(rc == TILEDB_OK);

  // Create attributes
  tiledb_attribute_t* a1;
  rc = tiledb_attribute_create(ctx_, &a1, "a1", TILEDB_INT32);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_compressor(ctx_, a1, TILEDB_BLOSC_LZ, -1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_cell_val_num(ctx_, a1, 1);
  CHECK(rc == TILEDB_OK);
  tiledb_attribute_t* a2;
  rc = tiledb_attribute_create(ctx_, &a2, "a2", TILEDB_CHAR);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_compressor(ctx_, a2, TILEDB_GZIP, -1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_cell_val_num(ctx_, a2, TILEDB_VAR_NUM);
  CHECK(rc == TILEDB_OK);
  tiledb_attribute_t* a3;
  rc = tiledb_attribute_create(ctx_, &a3, "a3", TILEDB_FLOAT32);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_compressor(ctx_, a3, TILEDB_ZSTD, -1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_attribute_set_cell_val_num(ctx_, a3, 2);
  CHECK(rc == TILEDB_OK);

  // Create array schmea
  tiledb_array_schema_t* array_schema;
  rc = tiledb_array_schema_create(ctx_, &array_schema, TILEDB_SPARSE);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_cell_order(ctx_, array_schema, TILEDB_ROW_MAJOR);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_tile_order(ctx_, array_schema, TILEDB_ROW_MAJOR);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_capacity(ctx_, array_schema, 2);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_set_domain(ctx_, array_schema, domain);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_add_attribute(ctx_, array_schema, a1);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_add_attribute(ctx_, array_schema, a2);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_schema_add_attribute(ctx_, array_schema, a3);
  CHECK(rc == TILEDB_OK);

  // Check array schema
  rc = tiledb_array_schema_check(ctx_, array_schema);
  CHECK(rc == TILEDB_OK);

  // Create array
  rc = tiledb_array_create(ctx_, SPARSE_ARRAY_NAME, array_schema);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_attribute_free(&a1);
  tiledb_attribute_free(&a2);
  tiledb_attribute_free(&a3);
  tiledb_dimension_free(&d1);
  tiledb_dimension_free(&d2);
  tiledb_domain_free(&domain);
  tiledb_array_schema_free(&array_schema);
}

void IncompleteFx::write_dense_full() {
  // Set attributes
  const char* attributes[] = {"a1", "a2", "a3"};

  // Prepare cell buffers
  // clang-format off
  int buffer_a1[] = {
      0,  1,  2,  3, 4,  5,  6,  7,
      8,  9,  10, 11, 12, 13, 14, 15
  };
  uint64_t buffer_a2[] = {
      0,  1,  3,  6, 10, 11, 13, 16,
      20, 21, 23, 26, 30, 31, 33, 36
  };
  char buffer_var_a2[] =
      "abbcccdddd"
      "effggghhhh"
      "ijjkkkllll"
      "mnnooopppp";
  float buffer_a3[] = {
      0.1f,  0.2f,  1.1f,  1.2f,  2.1f,  2.2f,  3.1f,  3.2f,
      4.1f,  4.2f,  5.1f,  5.2f,  6.1f,  6.2f,  7.1f,  7.2f,
      8.1f,  8.2f,  9.1f,  9.2f,  10.1f, 10.2f, 11.1f, 11.2f,
      12.1f, 12.2f, 13.1f, 13.2f, 14.1f, 14.2f, 15.1f, 15.2f,
  };
  void* buffers[] = { buffer_a1, buffer_a2, buffer_var_a2, buffer_a3 };
  uint64_t buffer_sizes[] =
  {
      sizeof(buffer_a1),
      sizeof(buffer_a2),
      sizeof(buffer_var_a2)-1,  // No need to store the last '\0' character
      sizeof(buffer_a3)
  };
  // clang-format on

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, DENSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_WRITE);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 3, buffers, buffer_sizes);
  CHECK(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  CHECK(rc == TILEDB_OK);

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  CHECK(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::write_sparse_full() {
  // Prepare cell buffers
  int buffer_a1[] = {0, 1, 2, 3, 4, 5, 6, 7};
  uint64_t buffer_a2[] = {0, 1, 3, 6, 10, 11, 13, 16};
  char buffer_var_a2[] = "abbcccddddeffggghhhh";
  float buffer_a3[] = {0.1f,
                       0.2f,
                       1.1f,
                       1.2f,
                       2.1f,
                       2.2f,
                       3.1f,
                       3.2f,
                       4.1f,
                       4.2f,
                       5.1f,
                       5.2f,
                       6.1f,
                       6.2f,
                       7.1f,
                       7.2f};
  uint64_t buffer_coords[] = {1, 1, 1, 2, 1, 4, 2, 3, 3, 1, 4, 2, 3, 3, 3, 4};
  void* buffers[] = {
      buffer_a1, buffer_a2, buffer_var_a2, buffer_a3, buffer_coords};
  uint64_t buffer_sizes[] = {
      sizeof(buffer_a1),
      sizeof(buffer_a2),
      sizeof(buffer_var_a2) - 1,  // No need to store the last '\0' character
      sizeof(buffer_a3),
      sizeof(buffer_coords)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, SPARSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  const char* attributes[] = {"a1", "a2", "a3", TILEDB_COORDS};
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_WRITE);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 4, buffers, buffer_sizes);
  CHECK(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  CHECK(rc == TILEDB_OK);

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  CHECK(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::remove_array(const std::string& array_name) {
  if (!is_array(array_name))
    return;

  CHECK(tiledb_object_remove(ctx_, array_name.c_str()) == TILEDB_OK);
}

void IncompleteFx::remove_dense_array() {
  remove_array(DENSE_ARRAY_NAME);
}

void IncompleteFx::remove_sparse_array() {
  remove_array(SPARSE_ARRAY_NAME);
}

bool IncompleteFx::is_array(const std::string& array_name) {
  tiledb_object_t type = TILEDB_INVALID;
  REQUIRE(tiledb_object_type(ctx_, array_name.c_str(), &type) == TILEDB_OK);
  return type == TILEDB_ARRAY;
}

void IncompleteFx::check_dense_incomplete() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 2, 1, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a1"};

  // Prepare the buffers that will store the result
  int buffer_a1[2];
  void* buffers[] = {buffer_a1};
  uint64_t buffer_sizes[] = {sizeof(buffer_a1)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, DENSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check status
  tiledb_query_status_t status;
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_INCOMPLETE);

  // Free/finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);

  // Check buffer
  int c_buffer_a1[2] = {0, 1};
  CHECK(!memcmp(buffer_a1, c_buffer_a1, sizeof(c_buffer_a1)));
  CHECK(buffer_sizes[0] == 2 * sizeof(int));
}

void IncompleteFx::check_dense_until_complete() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 2, 1, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a1"};

  // Prepare the buffers that will store the result
  int buffer_a1[2];
  void* buffers[] = {buffer_a1};
  uint64_t buffer_sizes[] = {sizeof(buffer_a1)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, DENSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check buffer
  int c_buffer_a1[2] = {0, 1};
  CHECK(!memcmp(buffer_a1, c_buffer_a1, sizeof(c_buffer_a1)));
  CHECK(buffer_sizes[0] == 2 * sizeof(int));

  // Check status
  tiledb_query_status_t status;
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_INCOMPLETE);

  // Resubmit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check status
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_COMPLETED);

  // Check new buffer contents
  int c_buffer_a1_2[2] = {2, 3};
  CHECK(!memcmp(buffer_a1, c_buffer_a1_2, sizeof(c_buffer_a1_2)));
  CHECK(buffer_sizes[0] == 2 * sizeof(int));

  // Free/finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::check_dense_unsplittable_overflow() {
  // Initialize a subarray
  const uint64_t subarray[] = {2, 2, 2, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a2"};

  // Prepare the buffers that will store the result
  uint64_t buffer_a2[1];
  char buffer_a2_var[1];
  void* buffers[] = {buffer_a2, buffer_a2_var};
  uint64_t buffer_sizes[] = {sizeof(buffer_a2), sizeof(buffer_a2_var)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, DENSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  CHECK(rc == TILEDB_ERR);

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::check_dense_unsplittable_complete() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 1, 2, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a2"};

  // Prepare the buffers that will store the result
  uint64_t buffer_a2[1];
  char buffer_a2_var[2];
  void* buffers[] = {buffer_a2, buffer_a2_var};
  uint64_t buffer_sizes[] = {sizeof(buffer_a2), sizeof(buffer_a2_var)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, DENSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  CHECK(rc == TILEDB_OK);

  // Check buffers
  char c_buffer_a2_var[2] = {'b', 'b'};
  CHECK(!memcmp(buffer_a2_var, c_buffer_a2_var, sizeof(c_buffer_a2_var)));

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::check_dense_reset_buffers() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 2, 1, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a1"};

  // Prepare the buffers that will store the result
  int buffer_a1[2];
  void* buffers[] = {buffer_a1};
  uint64_t buffer_sizes[] = {sizeof(buffer_a1)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, DENSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check buffer
  int c_buffer_a1[2] = {0, 1};
  CHECK(!memcmp(buffer_a1, c_buffer_a1, sizeof(c_buffer_a1)));
  CHECK(buffer_sizes[0] == 2 * sizeof(int));

  // Check status
  tiledb_query_status_t status;
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_INCOMPLETE);

  // Reset buffer sizes to something smaller
  uint64_t new_buffer_sizes[] = {1};
  rc = tiledb_query_reset_buffers(ctx_, query, buffers, new_buffer_sizes);
  CHECK(rc == TILEDB_ERR);

  // Reset buffer sizes to the original
  rc = tiledb_query_reset_buffers(ctx_, query, buffers, buffer_sizes);
  CHECK(rc == TILEDB_OK);

  // Resubmit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check status
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_COMPLETED);

  // Check new buffer contents
  int c_buffer_a1_2[2] = {2, 3};
  CHECK(!memcmp(buffer_a1, c_buffer_a1_2, sizeof(c_buffer_a1_2)));
  CHECK(buffer_sizes[0] == 2 * sizeof(int));

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::check_sparse_incomplete() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 2, 1, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a1"};

  // Prepare the buffers that will store the result
  int buffer_a1[1];
  void* buffers[] = {buffer_a1};
  uint64_t buffer_sizes[] = {sizeof(buffer_a1)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, SPARSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check status
  tiledb_query_status_t status;
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_INCOMPLETE);

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);

  // Check buffer
  int c_buffer_a1[1] = {0};
  CHECK(!memcmp(buffer_a1, c_buffer_a1, sizeof(c_buffer_a1)));
  CHECK(buffer_sizes[0] == sizeof(int));
}

void IncompleteFx::check_sparse_until_complete() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 2, 1, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a1"};

  // Prepare the buffers that will store the result
  int buffer_a1[1];
  void* buffers[] = {buffer_a1};
  uint64_t buffer_sizes[] = {sizeof(buffer_a1)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, SPARSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check status
  tiledb_query_status_t status;
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_INCOMPLETE);

  // Check buffer
  int c_buffer_a1[1] = {0};
  CHECK(!memcmp(buffer_a1, c_buffer_a1, sizeof(c_buffer_a1)));
  CHECK(buffer_sizes[0] == sizeof(int));

  // Resubmit the query
  rc = tiledb_query_submit(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Check status
  rc = tiledb_query_get_status(ctx_, query, &status);
  CHECK(rc == TILEDB_OK);
  CHECK(status == TILEDB_COMPLETED);

  // Check buffer
  c_buffer_a1[0] = 1;
  CHECK(!memcmp(buffer_a1, c_buffer_a1, sizeof(c_buffer_a1)));
  CHECK(buffer_sizes[0] == sizeof(int));

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::check_sparse_unsplittable_overflow() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 1, 2, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a2"};

  // Prepare the buffers that will store the result
  uint64_t buffer_a2[1];
  char buffer_a2_var[1];
  void* buffers[] = {buffer_a2, buffer_a2_var};
  uint64_t buffer_sizes[] = {sizeof(buffer_a2), sizeof(buffer_a2_var)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, SPARSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  CHECK(rc == TILEDB_ERR);

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

void IncompleteFx::check_sparse_unsplittable_complete() {
  // Initialize a subarray
  const uint64_t subarray[] = {1, 1, 2, 2};

  // Subset over a specific attribute
  const char* attributes[] = {"a2"};

  // Prepare the buffers that will store the result
  uint64_t buffer_a2[1];
  char buffer_a2_var[2];
  void* buffers[] = {buffer_a2, buffer_a2_var};
  uint64_t buffer_sizes[] = {sizeof(buffer_a2), sizeof(buffer_a2_var)};

  // Open array
  tiledb_array_t* array;
  int rc = tiledb_array_alloc(ctx_, SPARSE_ARRAY_NAME, &array);
  CHECK(rc == TILEDB_OK);
  rc = tiledb_array_open(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Create query
  tiledb_query_t* query;
  rc = tiledb_query_create(ctx_, &query, array, TILEDB_READ);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_buffers(
      ctx_, query, attributes, 1, buffers, buffer_sizes);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_subarray(ctx_, query, subarray);
  REQUIRE(rc == TILEDB_OK);
  rc = tiledb_query_set_layout(ctx_, query, TILEDB_GLOBAL_ORDER);
  REQUIRE(rc == TILEDB_OK);

  // Submit query
  rc = tiledb_query_submit(ctx_, query);
  CHECK(rc == TILEDB_OK);

  // Check buffers
  char c_buffer_a2_var[2] = {'b', 'b'};
  CHECK(!memcmp(buffer_a2_var, c_buffer_a2_var, sizeof(c_buffer_a2_var)));

  // Finalize query
  rc = tiledb_query_finalize(ctx_, query);
  REQUIRE(rc == TILEDB_OK);

  // Close array
  rc = tiledb_array_close(ctx_, array);
  CHECK(rc == TILEDB_OK);

  // Clean up
  tiledb_array_free(&array);
  tiledb_query_free(&query);
}

TEST_CASE_METHOD(
    IncompleteFx,
    "C API: Test incomplete read queries, dense",
    "[capi], [incomplete], [dense-incomplete]") {
  remove_dense_array();
  create_dense_array();
  write_dense_full();
  check_dense_incomplete();
  check_dense_until_complete();
  check_dense_unsplittable_overflow();
  check_dense_unsplittable_complete();
  check_dense_reset_buffers();
  remove_dense_array();
}

TEST_CASE_METHOD(
    IncompleteFx,
    "C API: Test incomplete read queries, sparse",
    "[capi], [incomplete], [incomplete]") {
  remove_sparse_array();
  create_sparse_array();
  write_sparse_full();
  check_sparse_incomplete();
  check_sparse_until_complete();
  check_sparse_unsplittable_overflow();
  check_sparse_unsplittable_complete();
  remove_sparse_array();
}