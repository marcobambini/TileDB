/**
 * @file   heap_memory.h
 *
 * @section LICENSE
 *
 * The MIT License
 *
 * @copyright Copyright (c) 2021 TileDB, Inc.
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
 * Defines TileDB-variants of dynamic (heap) memory allocation routines. When
 * the global `heap_profiler` is enabled, these routines will record memory
 * stats. Should allocation fail, stats will print and exit the program.
 */

#ifndef TILEDB_HEAP_MEMORY_H
#define TILEDB_HEAP_MEMORY_H

#include <cstdlib>
#include <memory>
#include <string>

#include "tiledb/common/heap_profiler.h"

namespace tiledb {
namespace common {

extern std::mutex __tdb_heap_mem_lock;

/** TileDB variant of `malloc`. */
void* tiledb_malloc(size_t size, const std::string& label);

/** TileDB variant of `calloc`. */
void* tiledb_calloc(size_t num, size_t size, const std::string& label);

/** TileDB variant of `realloc`. */
void* tiledb_realloc(void* p, size_t size, const std::string& label);

/** TileDB variant of `free`. */
void tiledb_free(void* p);

/** TileDB variant of `operator new`. */
template <typename T, typename... Args>
T* tiledb_new(const std::string& label, Args&&... args) {
  if (!heap_profiler.enabled()) {
    return new T(std::forward<Args>(args)...);
  }

  std::unique_lock<std::mutex> ul(__tdb_heap_mem_lock);

  T* const p = new T(std::forward<Args>(args)...);

  if (!p)
    heap_profiler.dump_and_terminate();

  heap_profiler.record_alloc(p, sizeof(T), label);

  return p;
}

/** TileDB variant of `operator delete`. */
template <typename T>
void tiledb_delete(T* const p) {
  if (!heap_profiler.enabled()) {
    delete p;
    return;
  }

  std::unique_lock<std::mutex> ul(__tdb_heap_mem_lock);

  delete p;
  heap_profiler.record_dealloc(p);
}

/** TileDB variant of `operator new[]`. */
template <typename T>
T* tiledb_new_array(const std::size_t size, const std::string& label) {
  if (!heap_profiler.enabled()) {
    return new T[size];
  }

  std::unique_lock<std::mutex> ul(__tdb_heap_mem_lock);

  T* const p = new T[size];

  if (!p)
    heap_profiler.dump_and_terminate();

  heap_profiler.record_alloc(p, sizeof(T) * size, label);

  return p;
}

/** TileDB variant of `operator delete[]`. */
template <typename T>
void tiledb_delete_array(T* const p) {
  if (!heap_profiler.enabled()) {
    delete[] p;
    return;
  }

  std::unique_lock<std::mutex> ul(__tdb_heap_mem_lock);

  delete[] p;
  heap_profiler.record_dealloc(p);
}

/** TileDB variant of `std::shared_ptr`. */
template <class T>
class tiledb_shared_ptr {
 public:
  tiledb_shared_ptr() = default;

  tiledb_shared_ptr(std::nullptr_t p)
      : sp_(p, tiledb_delete<T>) {
  }

  tiledb_shared_ptr(T* const p)
      : sp_(p, tiledb_delete<T>) {
  }

  tiledb_shared_ptr(const tiledb_shared_ptr& rhs)
      : sp_(rhs.sp_) {
  }

  tiledb_shared_ptr(tiledb_shared_ptr&& rhs)
      : sp_(std::move(rhs.sp_)) {
  }

  tiledb_shared_ptr& operator=(const tiledb_shared_ptr& rhs) {
    sp_ = rhs.sp_;
    return *this;
  }

  template <class Y>
  tiledb_shared_ptr& operator=(const tiledb_shared_ptr<Y>& rhs) {
    sp_ = rhs.inner_sp();
    return *this;
  }

  tiledb_shared_ptr& operator=(tiledb_shared_ptr&& rhs) {
    sp_ = std::move(rhs.sp_);
    return *this;
  }

  template <class Y>
  tiledb_shared_ptr& operator=(tiledb_shared_ptr<Y>&& rhs) {
    sp_ = std::move(rhs.inner_sp());
    return *this;
  }

  bool operator==(const tiledb_shared_ptr& rhs) const {
    return sp_ == rhs.sp_;
  }

  bool operator!=(const tiledb_shared_ptr& rhs) const {
    return sp_ != rhs.sp_;
  }

  void swap(tiledb_shared_ptr& rhs) noexcept {
    sp_.swap(rhs.sp_);
  }

  void reset(T* const p) {
    sp_.reset(p, tiledb_delete<T>);
  }

  T* get() const noexcept {
    return sp_.get();
  }

  T& operator*() const noexcept {
    return sp_.operator*();
  }

  T* operator->() const noexcept {
    return sp_.operator->();
  }

  explicit operator bool() const noexcept {
    return sp_ ? true : false;
  }

  long int use_count() const noexcept {
    return sp_.use_count();
  }

  std::shared_ptr<T> inner_sp() {
    return sp_;
  }

 private:
  std::shared_ptr<T> sp_;
};

/** TileDB variant of `std::unique_ptr`. */
template <class T>
struct TileDBUniquePtrDeleter {
  void operator()(T* const p) {
    tiledb_delete<T>(p);
  }
};
template <class T>
using tiledb_unique_ptr = std::unique_ptr<T, TileDBUniquePtrDeleter<T>>;

/** TileDB variant of `std::make_shared`. */
template <class T, typename... Args>
tiledb_shared_ptr<T> tiledb_make_shared(
    const std::string& label, Args&&... args) {
  return tiledb_shared_ptr<T>(
      tiledb_new<T>(label, std::forward<Args>(args)...));
}

}  // namespace common
}  // namespace tiledb

#define TILEDB_HEAP_MEM_LABEL \
  std::string(__FILE__) + std::string(":") + std::to_string(__LINE__)

#define tdb_malloc(size) \
  tiledb::common::tiledb_malloc(size, TILEDB_HEAP_MEM_LABEL)

#define tdb_calloc(num, size) \
  tiledb::common::tiledb_calloc(num, size, TILEDB_HEAP_MEM_LABEL)

#define tdb_realloc(p, size) \
  tiledb::common::tiledb_realloc(p, size, TILEDB_HEAP_MEM_LABEL)

#define tdb_free(p) tiledb::common::tiledb_free(p)

#ifdef _MSC_VER
#define tdb_new(T, ...) \
  tiledb::common::tiledb_new<T>(TILEDB_HEAP_MEM_LABEL, __VA_ARGS__)
#else
#define tdb_new(T, ...) \
  tiledb::common::tiledb_new<T>(TILEDB_HEAP_MEM_LABEL, ##__VA_ARGS__)
#endif

#define tdb_delete(p) tiledb::common::tiledb_delete(p);

#define tdb_new_array(T, size) \
  tiledb::common::tiledb_new_array<T>(size, TILEDB_HEAP_MEM_LABEL)

#define tdb_delete_array(p) tiledb::common::tiledb_delete_array(p)

#define tdb_shared_ptr tiledb::common::tiledb_shared_ptr

#define tdb_unique_ptr tiledb::common::tiledb_unique_ptr

#ifdef _MSC_VER
#define tdb_make_shared(T, ...) \
  tiledb::common::tiledb_make_shared<T>(TILEDB_HEAP_MEM_LABEL, __VA_ARGS__)
#else
#define tdb_make_shared(T, ...) \
  tiledb::common::tiledb_make_shared<T>(TILEDB_HEAP_MEM_LABEL, ##__VA_ARGS__)
#endif

#endif  // TILEDB_HEAP_MEMORY_H
