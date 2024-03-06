#pragma once

#include "pbrt/util/stack.h"
#include "pbrt/util/dynamic_array.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/shape.h"

struct BVHPrimitive {
    PBRT_GPU BVHPrimitive() : primitive_idx(SIZE_MAX) {}

    PBRT_GPU BVHPrimitive(size_t _primitive_idx, const Bounds3f &_bounds)
        : primitive_idx(_primitive_idx), bounds(_bounds),
          centroid(0.5 * (_bounds.p_min + _bounds.p_max)) {}

    PBRT_GPU bool operator==(const BVHPrimitive &bvh_primitive) const {
        return primitive_idx == bvh_primitive.primitive_idx && bounds == bvh_primitive.bounds &&
               centroid == bvh_primitive.centroid;
    }

    size_t primitive_idx;
    Bounds3f bounds;
    Point3f centroid;
};

struct BVHBuildNode {
  private:
    PBRT_GPU
    static int partition(BVHPrimitive *values, int _split_axis, double mid, int size) {
        int i = -1;
        int k = size;

        while (true) {
            do {
                k--;
            } while (values[k].centroid[_split_axis] > mid);

            do {
                i++;
            } while (values[i].centroid[_split_axis] <= mid);

            if (i < k) {
                auto temp = values[i];
                values[i] = values[k];
                values[k] = temp;
            } else {
                return k + 1;
            }
        }
    }

  public:
    Bounds3f bounds;
    BVHBuildNode *children[2] = {nullptr, nullptr};
    int split_axis = -1;
    int first_primitive_offset = -1;
    int primitive_num = -1;

    PBRT_GPU ~BVHBuildNode() {
        delete children[0];
        delete children[1];
    }

    PBRT_GPU void init_leaf(int first, int n, const Bounds3f &b) {
        first_primitive_offset = first;
        primitive_num = n;
        split_axis = -1;
        bounds = b;

        children[0] = nullptr;
        children[1] = nullptr;
    }

    PBRT_GPU void init_interior(int axis, BVHBuildNode *c0, BVHBuildNode *c1) {
        primitive_num = 0;
        first_primitive_offset = -1;
        split_axis = axis;
        bounds = c0->bounds + c1->bounds;

        children[0] = c0;
        children[1] = c1;
    }

    PBRT_GPU
    static BVHBuildNode *closure_build_leaf(const DynamicArray<const Shape *> &primitives,
                                            const BVHPrimitive *bvh_primitives,
                                            int n_bvh_primitives,
                                            DynamicArray<const Shape *> &ordered_primitives,
                                            const Bounds3f &bounds) {
        auto first_primitive_offset = ordered_primitives.size();
        for (int idx = 0; idx < n_bvh_primitives; ++idx) {
            auto primitive_idx = bvh_primitives[idx].primitive_idx;
            ordered_primitives.push(primitives[primitive_idx]);
        }

        auto node = new BVHBuildNode();
        node->init_leaf(first_primitive_offset, n_bvh_primitives, bounds);
        return node;
    }

    PBRT_GPU
    static BVHBuildNode *build_recursive(const DynamicArray<const Shape *> &primitives,
                                         BVHPrimitive *bvh_primitives, int n_bvh_primitives,
                                         DynamicArray<const Shape *> &ordered_primitives,
                                         int &node_count) {
        node_count += 1;
        auto full_bounds = bvh_primitives[0].bounds;
        for (int i = 1; i < n_bvh_primitives; ++i) {
            full_bounds += bvh_primitives[i].bounds;
        }

        if (full_bounds.surface_area() == 0.0 || n_bvh_primitives == 1) {
            return closure_build_leaf(primitives, bvh_primitives, n_bvh_primitives,
                                      ordered_primitives, full_bounds);
        }

        auto centroid_bounds = Bounds3f::empty();
        for (int i = 0; i < n_bvh_primitives; ++i) {
            centroid_bounds += bvh_primitives[i].centroid;
        }

        int split_axis = centroid_bounds.max_dimension();
        if (centroid_bounds.p_min[split_axis] == centroid_bounds.p_max[split_axis]) {
            // completely overlapped
            return closure_build_leaf(primitives, bvh_primitives, n_bvh_primitives,
                                      ordered_primitives, full_bounds);
        }

        auto mid_val =
            (centroid_bounds.p_min[split_axis] + centroid_bounds.p_max[split_axis]) / 2.0;

        int n_left = BVHBuildNode::partition(bvh_primitives, split_axis, mid_val, n_bvh_primitives);
        int n_right = n_bvh_primitives - n_left;

        if (n_left == 0 || n_right == 0) {
            return closure_build_leaf(primitives, bvh_primitives, n_bvh_primitives,
                                      ordered_primitives, full_bounds);
        }

        auto left_child = BVHBuildNode::build_recursive(primitives, bvh_primitives, n_left,
                                                        ordered_primitives, node_count);

        auto right_child = BVHBuildNode::build_recursive(primitives, bvh_primitives + n_left,
                                                         n_right, ordered_primitives, node_count);

        auto node = new BVHBuildNode();
        node->init_interior(split_axis, left_child, right_child);
        return node;
    }
};

#define ALIGNED_SIZE 64
struct alignas(ALIGNED_SIZE) LinearBVHNode {
    Bounds3f bounds;
    int offset = -1;
    /*
    for leaf:     primitives_offset
    for interior: second_child_offset;
    */
    uint16_t primitive_num = 65535;
    // 0 -> interior node
    uint8_t axis = 255;
    // interior node: xyz

  private:
    [[maybe_unused]] void check_alignment() const {
        constexpr int member_variables_size =
            sizeof(bounds) + sizeof(offset) + sizeof(primitive_num) + sizeof(axis);
        static_assert(ALIGNED_SIZE / member_variables_size == 1);
    }
};
#undef ALIGNED_SIZE

class BVH {
  public:
    PBRT_GPU explicit BVH(const DynamicArray<const Shape *> &primitives) {
        printf("building BVH for %d primitives\n", primitives.size());

        auto bvh_primitives = new BVHPrimitive[primitives.size()];
        for (int idx = 0; idx < primitives.size(); ++idx) {
            bvh_primitives[idx] = BVHPrimitive(idx, primitives[idx]->bounds());
        }

        int node_count = 0;
        ordered_primitives.reserve(primitives.size());

        auto root = BVHBuildNode::build_recursive(primitives, bvh_primitives, primitives.size(),
                                                  ordered_primitives, node_count);
        delete[] bvh_primitives;

        linear_bvh_nodes = new LinearBVHNode[node_count];

        int array_offset = 0;
        flatten_bvh(root, array_offset);
        delete root;

        printf("BVH built: (%d primitives, %d nodes)\n", primitives.size(), node_count);
    }

    PBRT_GPU ~BVH() {
        delete[] linear_bvh_nodes;
    }

    PBRT_GPU Bounds3f bounds() const {
        return linear_bvh_nodes[0].bounds;
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

            const auto node = &linear_bvh_nodes[current_node_idx];
            if (!node->bounds.fast_intersect(ray, t_max, inv_dir, dir_is_neg)) {
                continue;
            }

            if (node->primitive_num > 0) {
                for (int idx = node->offset; idx < node->offset + node->primitive_num; idx++) {
                    auto primitive = ordered_primitives[idx];
                    if (primitive->fast_intersect(ray, t_max)) {
                        return true;
                    }
                }
                continue;
            }

            // interior node
            // Put far BVH node on _nodesToVisit_ stack, advance to near node
            if (dir_is_neg[node->axis] > 0) {
                nodes_to_visit.push(current_node_idx + 1);
                nodes_to_visit.push(node->offset);
            } else {
                nodes_to_visit.push(node->offset);
                nodes_to_visit.push(current_node_idx + 1);
            }
        }
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray, double t_max) const {
        auto d = ray.d;
        auto inv_dir = Vector3f(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);
        int dir_is_neg[3] = {
            int(inv_dir.x < 0.0),
            int(inv_dir.y < 0.0),
            int(inv_dir.z < 0.0),
        };

        Stack<int, 128> nodes_to_visit;
        nodes_to_visit.push(0);
        auto best_t = t_max;
        std::optional<ShapeIntersection> best_intersection = std::nullopt;

        while (true) {
            if (nodes_to_visit.empty()) {
                break;
            }
            auto current_node_idx = nodes_to_visit.pop();

            const auto node = &linear_bvh_nodes[current_node_idx];
            if (!node->bounds.fast_intersect(ray, best_t, inv_dir, dir_is_neg)) {
                continue;
            }

            if (node->primitive_num > 0) {
                for (int idx = node->offset; idx < node->offset + node->primitive_num; idx++) {
                    auto primitive = ordered_primitives[idx];
                    auto intersection = primitive->intersect(ray, best_t);
                    if (!intersection) {
                        continue;
                    }

                    best_t = intersection->t_hit;
                    best_intersection = intersection;
                }
                continue;
            }

            // interior node
            // Put far BVH node on _nodesToVisit_ stack, advance to near node
            if (dir_is_neg[node->axis] > 0) {
                nodes_to_visit.push(current_node_idx + 1);
                nodes_to_visit.push(node->offset);
            } else {
                nodes_to_visit.push(node->offset);
                nodes_to_visit.push(current_node_idx + 1);
            }
        }

        return best_intersection;
    };

  private:
    PBRT_GPU int flatten_bvh(BVHBuildNode *node, int &array_offset) {
        LinearBVHNode *linearNode = &linear_bvh_nodes[array_offset];
        linearNode->bounds = node->bounds;
        int node_offset = array_offset++;
        if (node->primitive_num > 0) {
            // TODO: make sure node->primitive_num is smaller than 65536 (2^16)

            linearNode->offset = node->first_primitive_offset;
            linearNode->primitive_num = node->primitive_num;
            return node_offset;
        }

        linearNode->axis = node->split_axis;
        linearNode->primitive_num = 0;
        flatten_bvh(node->children[0], array_offset);
        linearNode->offset = flatten_bvh(node->children[1], array_offset);

        return node_offset;
    }

    LinearBVHNode *linear_bvh_nodes = nullptr;
    DynamicArray<const Shape *> ordered_primitives;
};
