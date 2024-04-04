#pragma once

#include <cassert>
#include <limits>
#include <vector>

#include "pbrt/base/shape.h"
#include "pbrt/euclidean_space/bounds3.h"

constexpr uint TREELET_MORTON_BITS_PER_DIMENSION = 10;
const uint BIT_LENGTH_TREELET_MASK = 18;
const uint MASK_OFFSET_BIT = TREELET_MORTON_BITS_PER_DIMENSION * 3 - BIT_LENGTH_TREELET_MASK;

constexpr uint MAX_TREELET_NUM = 262144;
/*
2 ^ 12 = 4096
2 ^ 15 = 32768
2 ^ 18 = 262144
2 ^ 21 = 2097152
*/

constexpr uint32_t TREELET_MASK = (MAX_TREELET_NUM - 1) << MASK_OFFSET_BIT;

constexpr uint MAX_PRIMITIVES_NUM_IN_LEAF = 1;

namespace {
[[maybe_unused]] void validate_treelet_num() {
    static_assert(MAX_TREELET_NUM == (1 << BIT_LENGTH_TREELET_MASK));
    static_assert(BIT_LENGTH_TREELET_MASK > 0 && BIT_LENGTH_TREELET_MASK < 32);
    static_assert(BIT_LENGTH_TREELET_MASK % 3 == 0);
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

    uint num_primitives;  // for leaf node
    uint right_child_idx; // for interior node

    // TODO: delete right_child_idx after rewriting top BVH building (right_idx = left_idx + 1)

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
    }

    PBRT_CPU_GPU
    void init_interior(uint8_t _axis, uint _left_child_idx, uint _right_child_offset,
                       const Bounds3f &_bounds) {
        left_child_idx = _left_child_idx;
        right_child_idx = _right_child_offset;
        num_primitives = 0;

        bounds = _bounds;
        axis = _axis;
    }
};

class HLBVH {
  public:
    void init(const Shape **_primitives, MortonPrimitive *gpu_morton_primitives,
              uint _n_primitives) {
        primitives = _primitives;
        morton_primitives = gpu_morton_primitives;
        num_total_primitives = _n_primitives;
    }

    PBRT_GPU void init_morton_primitives();

    PBRT_GPU void init_treelets(Treelet *treelets);

    PBRT_GPU void compute_morton_code(const Bounds3f &bounds_of_centroids);

    PBRT_GPU void collect_primitives_into_treelets(Treelet *treelets);

    PBRT_GPU void build_bottom_bvh(const BottomBVHArgs *bvh_args_array, uint array_length);

    PBRT_GPU bool fast_intersect(const Ray &ray, double t_max) const;

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray, double t_max) const;

    void build_bvh(std::vector<void *> &gpu_dynamic_pointers,
                   const std::vector<const Shape *> &gpu_primitives);

    const Shape **primitives;
    MortonPrimitive *morton_primitives;
    BVHBuildNode *build_nodes;
    uint num_total_primitives;

  private:
    uint build_top_bvh_for_treelets(uint num_treelets, const Treelet *treelets);

    uint recursive_build_top_bvh_for_treelets(const std::vector<uint> &treelet_indices,
                                              const Treelet *treelets, uint &build_node_count,
                                              uint depth, uint &max_depth);
    PBRT_GPU
    uint partition_morton_primitives(const uint start, const uint end,
                                     const uint8_t split_dimension, const double split_val);
};
