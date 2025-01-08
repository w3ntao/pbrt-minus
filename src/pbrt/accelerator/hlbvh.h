#pragma once

#include "pbrt/base/primitive.h"
#include "pbrt/base/shape.h"
#include "pbrt/euclidean_space/bounds3.h"
#include <atomic>
#include <cassert>
#include <vector>

class ThreadPool;

constexpr uint TREELET_MORTON_BITS_PER_DIMENSION = 10;
const uint BIT_LENGTH_OF_TREELET_MASK = 21;
const uint MASK_OFFSET_BIT = TREELET_MORTON_BITS_PER_DIMENSION * 3 - BIT_LENGTH_OF_TREELET_MASK;

constexpr uint MAX_TREELET_NUM = 1 << BIT_LENGTH_OF_TREELET_MASK;
/*
 total treelets:        ->    splits for each dimension:
 2 ^ 12 = 4096                2 ^ 4  = 16
 2 ^ 15 = 32768               2 ^ 5  = 32
 2 ^ 18 = 262144              2 ^ 6  = 64
 2 ^ 21 = 2097152             2 ^ 7  = 128
 2 ^ 24 = 16777216            2 ^ 8  = 256
 2 ^ 27 = 134217728           2 ^ 9  = 512
 2 ^ 30 = 1073741824          2 ^ 10 = 1024
*/

constexpr uint32_t TREELET_MASK = (MAX_TREELET_NUM - 1) << MASK_OFFSET_BIT;

constexpr uint MAX_PRIMITIVES_NUM_IN_LEAF = 1;

namespace {
[[maybe_unused]] void validate_treelet_num() {
    static_assert(BIT_LENGTH_OF_TREELET_MASK > 0 && BIT_LENGTH_OF_TREELET_MASK < 32);
    static_assert(BIT_LENGTH_OF_TREELET_MASK % 3 == 0);
}
} // namespace

struct MortonPrimitive {
    uint primitive_idx;
    uint32_t morton_code;
    Bounds3f bounds;
    Point3f centroid;
};

struct Treelet {
    uint first_primitive_offset;
    uint n_primitives;
    Bounds3f bounds;
};

struct BottomBVHArgs {
    uint build_node_idx;
    uint left_child_idx;
    bool expand_leaf;
};

struct BVHBuildNode {
    Bounds3f bounds;

    union {
        uint first_primitive_idx; // for leaf node
        uint left_child_idx;      // for interior node
    };

    uint num_primitives;
    // 0 for interior node
    // otherwise for leaf node

    /*
    uint8_t:         0 - 255
    uint16_t:        0 - 65535
    uint (uint32_t): 0 - 4294967295
    */

    uint8_t axis;

    PBRT_CPU_GPU
    inline bool is_leaf() const {
        return num_primitives > 0;
    }

    PBRT_CPU_GPU
    void init_leaf(uint _first_primitive_offset, uint _num_primitive, const Bounds3f &_bounds) {
        first_primitive_idx = _first_primitive_offset;
        num_primitives = _num_primitive;
        bounds = _bounds;
        axis = 255;
    }

    PBRT_CPU_GPU
    void init_interior(uint8_t _axis, uint _left_child_idx, const Bounds3f &_bounds) {
        left_child_idx = _left_child_idx;
        num_primitives = 0;

        bounds = _bounds;
        axis = _axis;
    }
};

class HLBVH {
  public:
    static const HLBVH *create(const std::vector<const Primitive *> &gpu_primitives,
                               std::vector<void *> &gpu_dynamic_pointers, ThreadPool &thread_pool);

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        if (build_nodes == nullptr) {
            return Bounds3f::empty();
        }

        return build_nodes[0].bounds;
    }

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    void build_bottom_bvh(const BottomBVHArgs *bvh_args_array, uint array_length);

  private:
    void init(const Primitive **_primitives, MortonPrimitive *gpu_morton_primitives) {
        primitives = _primitives;
        morton_primitives = gpu_morton_primitives;
        build_nodes = nullptr;
    }

    void build_bvh(const std::vector<const Primitive *> &gpu_primitives,
                   std::vector<void *> &gpu_dynamic_pointers, ThreadPool &thread_pool);

    uint build_top_bvh_for_treelets(const Treelet *treelets, uint num_dense_treelets,
                                    ThreadPool &thread_pool);

    void build_upper_sah(uint build_node_idx, std::vector<uint> treelet_indices,
                         const Treelet *treelets, std::atomic_int &node_count,
                         ThreadPool &thread_pool, bool spawn);

    PBRT_GPU
    uint partition_morton_primitives(uint start, uint end, uint8_t split_dimension,
                                     FloatType split_val);

    const Primitive **primitives;

    MortonPrimitive *morton_primitives;
    BVHBuildNode *build_nodes;
};
