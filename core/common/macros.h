/* Lightweight PROPERTY macro for project_6.
   Adapted from xllm/core/common/macros.h — stripped reflection. */
#pragma once

#include <utility>

#define PROPERTY(T, property)                                                  \
 public:                                                                       \
  [[nodiscard]] const T& property() const& noexcept { return property##_; }    \
  [[nodiscard]] T& property() & noexcept { return property##_; }               \
  auto property(const T& value) & -> decltype(*this) {                         \
    property##_ = value;                                                       \
    return *this;                                                              \
  }                                                                            \
  auto property(T&& value) & -> decltype(*this) {                             \
    property##_ = std::move(value);                                            \
    return *this;                                                              \
  }                                                                            \
  T property##_
