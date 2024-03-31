#pragma once

#include <cassert>
#include <limits>

#include "pbrt/base/shape.cuh"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/util/stack.cuh"

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
    PBRT_CPU_GPU void init(const Shape **_primitives, MortonPrimitive *gpu_morton_primitives,
                           uint _n_primitives) {
        primitives = _primitives;
        morton_primitives = gpu_morton_primitives;
        num_total_primitives = _n_primitives;
    }

    PBRT_GPU void init_morton_primitives() {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= num_total_primitives) {
            return;
        }

        morton_primitives[worker_idx].primitive_idx = worker_idx;

        const auto _bounds = primitives[worker_idx]->bounds();

        morton_primitives[worker_idx].bounds = _bounds;
        morton_primitives[worker_idx].centroid = _bounds.centroid();
    }

    PBRT_GPU void init_treelets(Treelet *treelets) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (worker_idx >= MAX_TREELET_NUM) {
            return;
        }

        treelets[worker_idx].first_primitive_offset = std::numeric_limits<uint>::max();
        treelets[worker_idx].n_primitives = 0;
    }

    PBRT_GPU void compute_morton_code(const Bounds3f bounds_of_centroids) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= num_total_primitives) {
            return;
        }

        constexpr int morton_scale = 1 << TREELET_MORTON_BITS_PER_DIMENSION;

        // compute morton code for each primitive
        auto centroid_offset = bounds_of_centroids.offset(morton_primitives[worker_idx].centroid);

        auto scaled_offset = centroid_offset * morton_scale;
        morton_primitives[worker_idx].morton_code = encode_morton3(
            uint32_t(scaled_offset.x), uint32_t(scaled_offset.y), uint32_t(scaled_offset.z));
    }

    PBRT_GPU void collect_primitives_into_treelets(Treelet *treelets) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= num_total_primitives) {
            return;
        }

        const uint start = worker_idx;
        uint32_t morton_start = morton_primitives[start].morton_code & TREELET_MASK;

        if (start == 0 ||
            morton_start != (morton_primitives[start - 1].morton_code & TREELET_MASK)) {
            // only if the worker starts from 0 or a gap will this function continue
        } else {
            return;
        }

        uint treelet_idx = morton_start >> MASK_OFFSET_BIT;

        uint start_primitive_idx = morton_primitives[start].primitive_idx;
        Bounds3f treelet_bounds = primitives[start_primitive_idx]->bounds();

        uint end = num_total_primitives;
        // if end doesn't match anything, it will be total_primitives
        for (uint idx = start + 1; idx < num_total_primitives; idx++) {
            uint32_t morton_end = morton_primitives[idx].morton_code & TREELET_MASK;

            if (morton_start != morton_end) {
                // discontinuity
                end = idx;
                break;
            }

            // exclude the "end" primitive in the treelet_bounds
            treelet_bounds += morton_primitives[idx].bounds;
        }

        treelets[treelet_idx].first_primitive_offset = start;
        treelets[treelet_idx].n_primitives = end - start;
        treelets[treelet_idx].bounds = treelet_bounds;
    }

    PBRT_GPU void build_bottom_bvh(const BottomBVHArgs *bvh_args_array, uint array_length) {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= array_length) {
            return;
        }

        const auto &args = bvh_args_array[worker_idx];

        if (!args.expand_leaf) {
            return;
        }

        const auto &node = build_nodes[args.build_node_idx];
        uint left_child_idx = args.left_child_idx + 0;
        uint right_child_idx = args.left_child_idx + 1;

        Bounds3f bounds_of_centroid;
        for (uint morton_idx = node.first_primitive_idx;
             morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
            bounds_of_centroid += morton_primitives[morton_idx].centroid;
        }

        auto split_dimension = bounds_of_centroid.max_dimension();
        auto split_val = bounds_of_centroid.centroid()[split_dimension];

        uint mid_idx = partition_morton_primitives(node.first_primitive_idx,
                                                   node.first_primitive_idx + node.num_primitives,
                                                   split_dimension, split_val);

        if (DEBUGGING) {
            bool kill_thread = false;

            if (mid_idx < node.first_primitive_idx ||
                mid_idx > node.first_primitive_idx + node.num_primitives) {
                printf("ERROR in partitioning at node[%u]: mid_idx out of bound\n",
                       args.build_node_idx);
                kill_thread = true;
            }

            for (uint morton_idx = node.first_primitive_idx; morton_idx < mid_idx; morton_idx++) {
                if (morton_primitives[morton_idx].centroid[split_dimension] >= split_val) {
                    printf("ERROR in partitioning (1st half) at node[%u], idx: %u\n",
                           args.build_node_idx, morton_idx);
                    kill_thread = true;
                }
            }

            for (uint morton_idx = mid_idx;
                 morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
                if (morton_primitives[morton_idx].centroid[split_dimension] < split_val) {
                    printf("ERROR in partitioning (2nd half) at node[%u], idx: %u\n",
                           args.build_node_idx, morton_idx);
                    kill_thread = true;
                }
            }

            if (kill_thread) {
                asm("trap;");
            }
        }

        if (mid_idx == node.first_primitive_idx ||
            mid_idx == node.first_primitive_idx + node.num_primitives) {
            // all primitives' centroids grouped either left or right half
            // there is no need to separate them

            build_nodes[left_child_idx].num_primitives = 0;
            build_nodes[right_child_idx].num_primitives = 0;

            return;
        }

        Bounds3f left_bounds;
        for (uint morton_idx = node.first_primitive_idx; morton_idx < mid_idx; morton_idx++) {
            left_bounds += morton_primitives[morton_idx].bounds;
        }

        Bounds3f right_bounds;
        for (uint morton_idx = mid_idx; morton_idx < node.first_primitive_idx + node.num_primitives;
             morton_idx++) {
            right_bounds += morton_primitives[morton_idx].bounds;
        }

        build_nodes[left_child_idx].init_leaf(node.first_primitive_idx,
                                              mid_idx - node.first_primitive_idx, left_bounds);
        build_nodes[right_child_idx].init_leaf(
            mid_idx, node.num_primitives - (mid_idx - node.first_primitive_idx), right_bounds);

        build_nodes[args.build_node_idx].init_interior(split_dimension, left_child_idx,
                                                       right_child_idx, left_bounds + right_bounds);
    }

    PBRT_GPU bool fast_intersect(const Ray &ray, double t_max) const {
        auto d = ray.d;
        auto inv_dir = Vector3f(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        int dir_is_neg[3] = {
            int(inv_dir.x < 0.0),
            int(inv_dir.y < 0.0),
            int(inv_dir.z < 0.0),
        };

        Stack<uint, 128> nodes_to_visit;
        nodes_to_visit.push(0);
        while (true) {
            if (nodes_to_visit.empty()) {
                return false;
            }
            auto current_node_idx = nodes_to_visit.pop();

            const auto node = build_nodes[current_node_idx];
            if (!node.bounds.fast_intersect(ray, t_max, inv_dir, dir_is_neg)) {
                continue;
            }

            if (node.is_leaf()) {
                for (uint morton_idx = node.first_primitive_idx;
                     morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
                    const uint primitive_idx = morton_primitives[morton_idx].primitive_idx;
                    auto const primitive = primitives[primitive_idx];

                    if (primitive->fast_intersect(ray, t_max)) {
                        return true;
                    }
                }
                continue;
            }

            if (dir_is_neg[node.axis] > 0) {
                nodes_to_visit.push(node.left_child_idx);
                nodes_to_visit.push(node.right_child_idx);
            } else {
                nodes_to_visit.push(node.right_child_idx);
                nodes_to_visit.push(node.left_child_idx);
            }
        }

        return false;
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray, double t_max) const {
        std::optional<ShapeIntersection> best_intersection = std::nullopt;
        auto best_t = t_max;

        auto d = ray.d;
        auto inv_dir = Vector3f(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        int dir_is_neg[3] = {
            int(inv_dir.x < 0.0),
            int(inv_dir.y < 0.0),
            int(inv_dir.z < 0.0),
        };

        Stack<uint, 128> nodes_to_visit;
        nodes_to_visit.push(0);

        while (true) {
            if (nodes_to_visit.empty()) {
                break;
            }
            auto current_node_idx = nodes_to_visit.pop();

            const auto node = build_nodes[current_node_idx];
            if (!node.bounds.fast_intersect(ray, best_t, inv_dir, dir_is_neg)) {
                continue;
            }

            if (node.is_leaf()) {
                for (uint morton_idx = node.first_primitive_idx;
                     morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
                    const uint primitive_idx = morton_primitives[morton_idx].primitive_idx;
                    auto const primitive = primitives[primitive_idx];

                    auto intersection = primitive->intersect(ray, best_t);
                    if (!intersection) {
                        continue;
                    }

                    best_t = intersection->t_hit;
                    best_intersection = intersection;
                }
                continue;
            }

            if (dir_is_neg[node.axis] > 0) {
                nodes_to_visit.push(node.left_child_idx);
                nodes_to_visit.push(node.right_child_idx);
            } else {
                nodes_to_visit.push(node.right_child_idx);
                nodes_to_visit.push(node.left_child_idx);
            }
        }

        return best_intersection;
    };

    uint build_top_bvh_for_treelets(uint num_treelets, const Treelet *treelets) {
        std::vector<uint> treelet_indices;
        for (uint idx = 0; idx < num_treelets; idx++) {
            treelet_indices.push_back(idx);
        }

        uint build_node_count = 0;
        uint max_depth = 0;
        recursive_build_top_bvh_for_treelets(treelet_indices, treelets, build_node_count, 0,
                                             max_depth);

        printf("HLBVH: top BVH nodes: %u, max depth: %u\n", build_node_count, max_depth);

        return build_node_count;
    }

    const Shape **primitives;
    MortonPrimitive *morton_primitives;
    BVHBuildNode *build_nodes;
    uint num_total_primitives;

  private:
    uint recursive_build_top_bvh_for_treelets(const std::vector<uint> &treelet_indices,
                                              const Treelet *treelets, uint &build_node_count,
                                              uint depth, uint &max_depth) {
        max_depth = std::max(depth, max_depth);

        uint current_build_node_idx = build_node_count;
        build_node_count += 1;

        if (treelet_indices.size() == 1) {
            uint treelet_idx = treelet_indices[0];
            const auto &current_treelet = treelets[treelet_idx];

            build_nodes[current_build_node_idx].init_leaf(current_treelet.first_primitive_offset,
                                                          current_treelet.n_primitives,
                                                          current_treelet.bounds);

            return current_build_node_idx;
        }

        Bounds3f _bounds_of_centroids;
        for (const auto treelet_idx : treelet_indices) {
            _bounds_of_centroids += treelets[treelet_idx].bounds.centroid();
        }
        uint8_t split_axis = _bounds_of_centroids.max_dimension();
        auto split_val = _bounds_of_centroids.centroid()[split_axis];

        std::vector<uint> left_indices;
        std::vector<uint> right_indices;

        for (const auto idx : treelet_indices) {
            if (treelets[idx].bounds.centroid()[split_axis] < split_val) {
                left_indices.push_back(idx);
            } else {
                right_indices.push_back(idx);
            }
        }

        if (left_indices.empty() || right_indices.empty()) {
            printf("ERROR: conflicted treelets centroid\n");
            printf("ERROR: each treelets are independent that they shouldn't overlap\n");
            exit(1);
        }

        uint left_build_node_idx = recursive_build_top_bvh_for_treelets(
            left_indices, treelets, build_node_count, depth + 1, max_depth);
        uint right_build_node_idx = recursive_build_top_bvh_for_treelets(
            right_indices, treelets, build_node_count, depth + 1, max_depth);

        auto bounds_combined =
            build_nodes[left_build_node_idx].bounds + build_nodes[right_build_node_idx].bounds;

        build_nodes[current_build_node_idx].init_interior(split_axis, left_build_node_idx,
                                                          right_build_node_idx, bounds_combined);

        return current_build_node_idx;
    }

    PBRT_GPU
    uint partition_morton_primitives(const uint start, const uint end,
                                     const uint8_t split_dimension, const double split_val) {
        // taken and modified from
        // https://users.cs.duke.edu/~reif/courses/alglectures/littman.lectures/lect05/node27.html

        uint left = start;
        uint right = end - 1;

        while (true) {
            while (morton_primitives[right].centroid[split_dimension] >= split_val &&
                   right > start) {
                right--;
            }

            while (morton_primitives[left].centroid[split_dimension] < split_val &&
                   left < end - 1) {
                left++;
            }

            if (left < right) {
                const auto temp = morton_primitives[left];
                morton_primitives[left] = morton_primitives[right];
                morton_primitives[right] = temp;
                continue;
            }

            if (left == start && right == start) {
                return start;
            }

            return right + 1;
        }
    }
};
