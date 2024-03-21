#pragma once

#include <cassert>

#include "pbrt/base/shape.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/util/stack.h"

constexpr int TREELET_MORTON_BITS_PER_DIMENSION = 10;
const int BIT_LENGTH_TREELET_MASK = 15;
const int MASK_OFFSET_BIT = TREELET_MORTON_BITS_PER_DIMENSION * 3 - BIT_LENGTH_TREELET_MASK;

constexpr int MAX_TREELET_NUM = 32768;
/*
2 ^ 12 = 4096
2 ^ 15 = 32768
*/

const uint32_t TREELET_MASK = (MAX_TREELET_NUM - 1) << MASK_OFFSET_BIT;

namespace {
[[maybe_unused]] void validate_treelet_num() {
    static_assert(MAX_TREELET_NUM == (1 << BIT_LENGTH_TREELET_MASK));
    static_assert(BIT_LENGTH_TREELET_MASK > 0 && BIT_LENGTH_TREELET_MASK < 32 &&
                  BIT_LENGTH_TREELET_MASK % 3 == 0);
}
} // namespace

struct MortonPrimitive {
    int primitive_idx;
    uint32_t morton_code;
};

struct BVHPrimitive {
    PBRT_GPU bool operator==(const BVHPrimitive &bvh_primitive) const {
        return primitive_idx == bvh_primitive.primitive_idx && bounds == bvh_primitive.bounds;
    }

    size_t primitive_idx;
    Bounds3f bounds;
    Point3f centroid;
};

struct Treelet {
    int start_idx = -1;
    int n_primitives = 0;
    Bounds3f bounds;
};

struct BVHBuildNodeForTreelet {
    // TODO: rewrite BVHBuildNodeForTreelet
    Bounds3f bounds;
    int treelet_idx;
    int left_child;
    int right_child;
    int split_axis;

    void init_leaf(int _treelet_idx, const Bounds3f &_bounds) {
        treelet_idx = _treelet_idx;
        bounds = _bounds;
        split_axis = -1;
        left_child = -1;
        right_child = -1;
    }

    void init_interior(int axis, int _left_child, int _right_child, const Bounds3f &_bounds) {
        split_axis = axis;
        left_child = _left_child;
        right_child = _right_child;
        treelet_idx = -1;
        bounds = _bounds;
    }
};

class HLBVH {
  public:
    PBRT_CPU_GPU void init(const Shape **_primitives, BVHPrimitive *gpu_bvh_primitives,
                           MortonPrimitive *gpu_morton_primitives, int _n_primitives) {
        primitives = _primitives;
        bvh_primitives = gpu_bvh_primitives;
        morton_primitives = gpu_morton_primitives;
        total_primitives = _n_primitives;

        bounds_of_centroids = Bounds3f::empty();
    }

    PBRT_GPU void init_bvh_primitives_and_treelets() {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        // worker_idx should cover full range of GPU_SHAPE_NUM and MAX_TREELET_NUM
        if (worker_idx < total_primitives) {
            bvh_primitives[worker_idx].primitive_idx = worker_idx;
            bvh_primitives[worker_idx].bounds = primitives[worker_idx]->bounds();
            bvh_primitives[worker_idx].centroid = bvh_primitives[worker_idx].bounds.centroid();
        }

        if (worker_idx < MAX_TREELET_NUM) {
            treelets[worker_idx].start_idx = -1;
            treelets[worker_idx].n_primitives = 0;
        }
    }

    PBRT_GPU void compute_bounds_of_centroids() {
        // TODO: paralellize this later

        bounds_of_centroids = Bounds3f::empty();
        for (int idx = 0; idx < total_primitives; idx++) {
            bounds_of_centroids += bvh_primitives[idx].bounds.centroid();
        }
    }

    PBRT_GPU void compute_morton_code() {
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        // worker_idx should cover full range of GPU_SHAPE_NUM and TOP_TREELET_MAX_NUM

        constexpr int morton_scale = 1 << TREELET_MORTON_BITS_PER_DIMENSION;

        if (worker_idx < total_primitives) {
            // compute morton code for each primitive
            morton_primitives[worker_idx].primitive_idx = bvh_primitives[worker_idx].primitive_idx;
            auto centroid_offset = bounds_of_centroids.offset(bvh_primitives[worker_idx].centroid);

            auto scaled_offset = centroid_offset * morton_scale;
            morton_primitives[worker_idx].morton_code = encode_morton3(
                uint32_t(scaled_offset.x), uint32_t(scaled_offset.y), uint32_t(scaled_offset.z));
        }
    }

    PBRT_GPU void build_treelets() {
        // TODO: progress 2024/03/15 building top_treelets
        const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (worker_idx >= total_primitives) {
            return;
        }

        const uint start = worker_idx;
        uint32_t bit_start = morton_primitives[start].morton_code & TREELET_MASK;

        if (start == 0 || bit_start != (morton_primitives[start - 1].morton_code & TREELET_MASK)) {
            // find the gap
            int primitive_idx = morton_primitives[start].primitive_idx;
            Bounds3f treelet_bounds = primitives[primitive_idx]->bounds();

            for (uint end = start + 1; end <= total_primitives; end++) {
                if (end < total_primitives) {
                    int end_primitive_idx = morton_primitives[end].primitive_idx;
                    treelet_bounds += primitives[end_primitive_idx]->bounds();
                }

                if (end == total_primitives ||
                    bit_start != (morton_primitives[end].morton_code & TREELET_MASK)) {

                    uint treelet_idx = bit_start >> MASK_OFFSET_BIT;

                    treelets[treelet_idx].start_idx = start;
                    treelets[treelet_idx].n_primitives = end - start;
                    treelets[treelet_idx].bounds = treelet_bounds;
                    break;
                }
            }
        }
    }

    int recursive_build_bvh_for_treelets(const std::vector<int> &treelet_indices,
                                         const Bounds3f &_full_bounds, int &treelet_node_count,
                                         int level, int &max_level) {
        max_level = std::max(level, max_level);

        int current_build_node_idx = treelet_node_count;
        treelet_node_count += 1;

        if (treelet_indices.size() == 1) {
            build_nodes_for_treelets[current_build_node_idx].init_leaf(treelet_indices[0],
                                                                       _full_bounds);
            return current_build_node_idx;
        }

        Bounds3f _bounds_of_centroids;
        for (const auto treelet_idx : treelet_indices) {
            _bounds_of_centroids += treelets[treelet_idx].bounds;
        }
        auto split_axis = _bounds_of_centroids.max_dimension();
        auto split_val = _bounds_of_centroids.centroid()[split_axis];

        std::vector<int> left_indices;
        std::vector<int> right_indices;

        Bounds3f left_bounds;
        Bounds3f right_bounds;
        for (const int treelet_idx : treelet_indices) {
            if (treelets[treelet_idx].bounds.centroid()[split_axis] < split_val) {
                left_indices.push_back(treelet_idx);
                left_bounds += treelets[treelet_idx].bounds;
            } else {
                right_indices.push_back(treelet_idx);
                right_bounds += treelets[treelet_idx].bounds;
            }
        }

        if (left_indices.empty() || right_indices.empty()) {
            // TODO: progress 2024/03/19 rewrite this to handle multiple treelets in one node

            auto left_treelet_idx = treelet_indices[0];

            std::vector<int> _right_indices = {treelet_indices.begin() + 1, treelet_indices.end()};

            Bounds3f _right_bounds;
            for (const auto idx : _right_indices) {
                _right_bounds += treelets[idx].bounds;
            }

            std::vector<int> _left_indices = {left_treelet_idx};

            int left_build_node_idx = recursive_build_bvh_for_treelets(
                _left_indices, Bounds3f(treelets[left_treelet_idx].bounds), treelet_node_count,
                level + 1, max_level);

            int right_build_node_idx = recursive_build_bvh_for_treelets(
                _right_indices, _right_bounds, treelet_node_count, level + 1, max_level);

            build_nodes_for_treelets[current_build_node_idx].init_interior(
                split_axis, left_build_node_idx, right_build_node_idx, _full_bounds);

            return current_build_node_idx;
        }

        int left_build_node_idx = recursive_build_bvh_for_treelets(
            left_indices, left_bounds, treelet_node_count, level + 1, max_level);
        int right_build_node_idx = recursive_build_bvh_for_treelets(
            right_indices, right_bounds, treelet_node_count, level + 1, max_level);

        build_nodes_for_treelets[current_build_node_idx].init_interior(
            split_axis, left_build_node_idx, right_build_node_idx, _full_bounds);

        return current_build_node_idx;
    }

    void build_bvh_for_treelets() {
        std::vector<int> treelet_indices;
        Bounds3f full_bounds;
        for (int idx = 0; idx < MAX_TREELET_NUM; idx++) {
            int n_primitives = treelets[idx].n_primitives;
            if (n_primitives <= 0) {
                continue;
            }
            full_bounds += treelets[idx].bounds;

            treelet_indices.push_back(idx);
        }

        int treelet_node_count = 0;
        int max_level = 0;
        int root_idx = recursive_build_bvh_for_treelets(treelet_indices, full_bounds,
                                                        treelet_node_count, 0, max_level);
        assert(root_idx == 0);

        printf("HLBVH: %d nodes built for %zu treelets (max level: %d)\n", treelet_node_count,
               treelet_indices.size(), max_level);
    }

    PBRT_GPU bool fast_intersect(const Ray &ray, double t_max) const {
        auto d = ray.d;
        auto inv_dir = Vector3f(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        int dir_is_neg[3] = {
            int(inv_dir.x < 0.0),
            int(inv_dir.y < 0.0),
            int(inv_dir.z < 0.0),
        };

        Stack<int, 128> nodes_to_visit;
        nodes_to_visit.push(0);
        while (true) {
            if (nodes_to_visit.empty()) {
                return false;
            }
            auto current_node_idx = nodes_to_visit.pop();

            const auto node = build_nodes_for_treelets[current_node_idx];
            if (!node.bounds.fast_intersect(ray, t_max, inv_dir, dir_is_neg)) {
                continue;
            }

            if (node.treelet_idx >= 0) {
                const auto &_treelet = treelets[node.treelet_idx];
                for (int offset = 0; offset < _treelet.n_primitives; offset++) {
                    int morton_idx = offset + _treelet.start_idx;
                    const int primitive_idx = morton_primitives[morton_idx].primitive_idx;
                    auto const primitive = primitives[primitive_idx];

                    if (primitive->fast_intersect(ray, t_max)) {
                        return true;
                    }
                }
                continue;
            }

            if (dir_is_neg[node.split_axis] > 0) {
                nodes_to_visit.push(node.left_child);
                nodes_to_visit.push(node.right_child);
            } else {
                nodes_to_visit.push(node.right_child);
                nodes_to_visit.push(node.left_child);
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

        Stack<int, 128> nodes_to_visit;
        nodes_to_visit.push(0);

        while (true) {
            if (nodes_to_visit.empty()) {
                break;
            }
            auto current_node_idx = nodes_to_visit.pop();

            const auto node = build_nodes_for_treelets[current_node_idx];
            if (!node.bounds.fast_intersect(ray, best_t, inv_dir, dir_is_neg)) {
                continue;
            }

            if (node.treelet_idx > 0) {
                const auto &_treelet = treelets[node.treelet_idx];
                for (int offset = 0; offset < _treelet.n_primitives; offset++) {
                    int morton_idx = offset + _treelet.start_idx;
                    const int primitive_idx = morton_primitives[morton_idx].primitive_idx;
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

            if (dir_is_neg[node.split_axis] > 0) {
                nodes_to_visit.push(node.left_child);
                nodes_to_visit.push(node.right_child);
            } else {
                nodes_to_visit.push(node.right_child);
                nodes_to_visit.push(node.left_child);
            }
        }

        return best_intersection;
    };

    const Shape **primitives;

    BVHPrimitive *bvh_primitives = nullptr;
    MortonPrimitive *morton_primitives = nullptr;

    Treelet treelets[MAX_TREELET_NUM];
    // TODO: collect filled treelets into another array
    BVHBuildNodeForTreelet *build_nodes_for_treelets = nullptr;

    int total_primitives;
    Bounds3f bounds_of_centroids; // bounds: bounds of centroids fro all primitives
};
