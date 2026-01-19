//===- cache_info.hpp - CPU Cache Information Query -------------*- C++ -*-===//
//
// Query CPU cache sizes for optimal tiling
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CACHE_INFO_HPP
#define MLIR_CACHE_INFO_HPP

#include <cstdint>
#include <utility>

namespace mlir {
namespace cherry {

struct CacheInfo
{
    uint64_t l1_cache_size = 32 * 1024;         // 32KB default
    uint64_t l2_cache_size = 256 * 1024;        // 256KB default
    uint64_t l3_cache_size = 8 * 1024 * 1024;   // 8MB default

    /// Query actual CPU cache sizes from the system
    CacheInfo();
    std::pair<int, int> computeMatmulTileSizes(int element_size) const;
};

}   // namespace cherry
}   // namespace mlir

#endif   // MLIR_CACHE_INFO_HPP
