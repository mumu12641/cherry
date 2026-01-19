//===- cache_info.cpp - CPU Cache Information Query ---------------------===//

#include "Utils/Cache.h"

#include <cmath>
#include <fstream>
#include <llvm/Support/raw_ostream.h>
#include <string>
#include <utility>

namespace mlir {
namespace cherry {

// Helper to parse cache size from sysfs (e.g., "32K" → 32768)
static uint64_t parseCacheSize(const std::string& str)
{
    if (str.empty()) return 0;

    uint64_t value = 0;
    size_t   i     = 0;
    while (i < str.size() && std::isdigit(str[i])) {
        value = value * 10 + (str[i] - '0');
        i++;
    }

    // Check for K/M suffix
    if (i < str.size()) {
        char suffix = str[i];
        if (suffix == 'K' || suffix == 'k') {
            value *= 1024;
        }
        else if (suffix == 'M' || suffix == 'm') {
            value *= 1024 * 1024;
        }
    }

    return value;
}

// Read cache size from Linux sysfs
static uint64_t readCacheSizeFromSysfs(int level, const char* type = "d")
{
    // /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/size
    // index0 = L1 data cache
    // index1 = L1 instruction cache
    // index2 = L2 cache
    // index3 = L3 cache

    int index = (level == 1) ? 0 : (level == 2) ? 2 : 3;

    std::string path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/size";

    std::ifstream file(path);
    if (!file.is_open()) {
        return 0;
    }

    std::string size_str;
    std::getline(file, size_str);
    return parseCacheSize(size_str);
}

CacheInfo::CacheInfo()
{
    // Try to read from Linux sysfs
    uint64_t l1 = readCacheSizeFromSysfs(1);
    uint64_t l2 = readCacheSizeFromSysfs(2);
    uint64_t l3 = readCacheSizeFromSysfs(3);

    if (l1 > 0) this->l1_cache_size = l1;
    if (l2 > 0) this->l2_cache_size = l2;
    if (l3 > 0) this->l3_cache_size = l3;
}

static int findOptimalTileSize(uint64_t cacheSizeBytes, double safetyFactor, int elementSize)
{
    uint64_t usableCache = static_cast<uint64_t>(cacheSizeBytes * safetyFactor);

    std::vector<int> candidates = {16, 32, 64, 128, 256, 512};
    int              bestT      = 16;

    for (int t : candidates) {
        uint64_t requiredMemory = 3ULL * (t * t) * elementSize;
        double   utilization    = (double)requiredMemory / cacheSizeBytes * 100.0;

        if (requiredMemory <= usableCache) {
            bestT = t;
        }
        else {
            break;
        }
    }
    return bestT;
}

std::pair<int, int> CacheInfo::computeMatmulTileSizes(int element_size) const
{
    int raw_l1  = findOptimalTileSize(l1_cache_size, 0.4, element_size);
    int l1_tile = std::max(4, std::min(32, raw_l1));

    int raw_l2  = findOptimalTileSize(l2_cache_size, 0.8, element_size);
    int l2_tile = std::max(16, std::min(128, raw_l2));

    // int raw_l3 = findOptimalTileSize(l3_cache_size, 0.9, element_size);
    // l3_tile    = std::max(64, std::min(512, raw_l3));

    // llvm::outs() << "[Tile Sizes] L1_Tile: " << l1_tile << ", L2_Tile: " << l2_tile
    //              << ", L3_Tile: " << l3_tile << "\n";
    return {l1_tile, l2_tile};
}

}   // namespace cherry
}   // namespace mlir
