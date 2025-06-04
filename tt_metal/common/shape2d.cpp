// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <array>
#include <cstddef>
#include <ostream>
#include <utility>

#include "shape2d.hpp"

namespace tt::tt_metal {

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::Shape2D& size) {
    os << "(" << size.height() << ", " << size.width() << ")";
    return os;
}

}  // namespace tt::tt_metal
