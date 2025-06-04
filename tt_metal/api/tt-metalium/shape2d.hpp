// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <tuple>
#include <ostream>

namespace tt::tt_metal {

// Simplified 2D shape for cases that fundamentally require 2 dimensions (e.g. core grid).
class Shape2D final {
public:
    Shape2D(std::size_t height, std::size_t width) : height_(height), width_(width) {}
    Shape2D(const std::pair<std::size_t, std::size_t>& size) : Shape2D(size.first, size.second) {}
    Shape2D(const std::array<std::size_t, 2>& size) : Shape2D(size[0], size[1]) {}
    Shape2D(const std::array<std::uint32_t, 2>& size) : Shape2D(size[0], size[1]) {}

    operator std::pair<std::size_t, std::size_t>() const { return {height_, width_}; }
    operator std::array<std::size_t, 2>() const { return {height_, width_}; }
    operator std::array<std::uint32_t, 2>() const {
        return {static_cast<uint32_t>(height_), static_cast<uint32_t>(width_)};
    }

    Shape2D operator*(std::size_t scalar) const { return Shape2D(height_ * scalar, width_ * scalar); }

    bool operator==(const Shape2D& rhs) const { return height_ == rhs.height_ && width_ == rhs.width_; }

    [[nodiscard]] std::size_t height() const { return height_; }
    [[nodiscard]] std::size_t width() const { return width_; }

    static constexpr auto attribute_names = std::forward_as_tuple("height", "width");
    auto attribute_values() const { return std::forward_as_tuple(height_, width_); }

    // Enable structured bindings.
    template <std::size_t I>
    size_t get() const {
        if constexpr (I == 0) {
            return height_;
        } else if constexpr (I == 1) {
            return width_;
        } else {
            static_assert(I < 2, "Invalid index for Shape2D");
        }
    }

private:
    std::size_t height_ = 0;
    std::size_t width_ = 0;
};

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Shape2D& size);

}  // namespace tt::tt_metal

namespace std {
template <>
struct tuple_size<tt::tt_metal::Shape2D> : std::integral_constant<size_t, 2> {};

template <std::size_t I>
struct tuple_element<I, tt::tt_metal::Shape2D> {
    using type = std::size_t;
};
}  // namespace std
