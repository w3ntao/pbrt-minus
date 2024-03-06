#pragma once

#include "pbrt/util/stack.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/shape.h"

struct BVHPrimitive {
    PBRT_GPU BVHPrimitive() : primitive_idx(SIZE_MAX) {}

    PBRT_GPU BVHPrimitive(size_t _primitive_idx, const Bounds3f &_bounds)
        : primitive_idx(_primitive_idx), bounds(_bounds),
          centroid(0.5 * (_bounds.p_min + _bounds.p_max)) {}

    size_t primitive_idx;
    Bounds3f bounds;
    Point3f centroid;
};

struct BVHBuildNode {
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
    static BVHBuildNode *closure_build_leaf(const Shape *const *primitives,
                                            const BVHPrimitive bvh_primitives[],
                                            int n_bvh_primitives, const Shape **ordered_primitives,
                                            int &n_ordered_primitives, const Bounds3f &bounds) {
        auto first_primitive_offset = n_ordered_primitives;
        for (int idx = 0; idx < n_bvh_primitives; ++idx) {
            auto primitive_idx = bvh_primitives[idx].primitive_idx;
            ordered_primitives[n_ordered_primitives] = primitives[primitive_idx];
            n_ordered_primitives += 1;
        }

        auto node = new BVHBuildNode();
        node->init_leaf(first_primitive_offset, n_bvh_primitives, bounds);
        return node;
    }

    PBRT_GPU
    static BVHBuildNode *build_recursive(const Shape *const *primitives,
                                         const BVHPrimitive bvh_primitives[], int n_bvh_primitives,
                                         const Shape **ordered_primitives,
                                         int &n_ordered_primitives, int &node_count) {
        // TODO: rewrite this interface with DynamicArray

        node_count += 1;
        auto full_bounds = bvh_primitives[0].bounds;
        for (int i = 1; i < n_bvh_primitives; ++i) {
            full_bounds += bvh_primitives[i].bounds;
        }

        if (full_bounds.surface_area() == 0.0 || n_bvh_primitives == 1) {
            return closure_build_leaf(primitives, bvh_primitives, n_bvh_primitives,
                                      ordered_primitives, n_ordered_primitives, full_bounds);
        }

        auto centroid_bounds = Bounds3f::empty();
        for (int i = 0; i < n_bvh_primitives; ++i) {
            centroid_bounds += bvh_primitives[i].centroid;
        }

        int split_axis = centroid_bounds.max_dimension();
        if (centroid_bounds.p_min[split_axis] == centroid_bounds.p_max[split_axis]) {
            // completely overlapped
            return closure_build_leaf(primitives, bvh_primitives, n_bvh_primitives,
                                      ordered_primitives, n_ordered_primitives, full_bounds);
        }

        auto mid_val =
            (centroid_bounds.p_min[split_axis] + centroid_bounds.p_max[split_axis]) / 2.0;

        // TODO: rewrite left_primitives and right_primitives with DynamicArray
        auto left_primitives = new BVHPrimitive[n_bvh_primitives];
        auto right_primitives = new BVHPrimitive[n_bvh_primitives];

        int n_left = 0;
        int n_right = 0;
        for (int i = 0; i < n_bvh_primitives; ++i) {
            if (bvh_primitives[i].centroid[split_axis] <= mid_val) {
                left_primitives[n_left] = bvh_primitives[i];
                n_left += 1;
            } else {
                right_primitives[n_right] = bvh_primitives[i];
                n_right += 1;
            }
        }

        if (n_left == 0 || n_right == 0) {
            return closure_build_leaf(primitives, bvh_primitives, n_bvh_primitives,
                                      ordered_primitives, n_ordered_primitives, full_bounds);
        }

        auto left_child =
            BVHBuildNode::build_recursive(primitives, left_primitives, n_left, ordered_primitives,
                                          n_ordered_primitives, node_count);
        delete left_primitives;

        auto right_child =
            BVHBuildNode::build_recursive(primitives, right_primitives, n_right, ordered_primitives,
                                          n_ordered_primitives, node_count);

        delete right_primitives;

        auto node = new BVHBuildNode();
        node->init_interior(split_axis, left_child, right_child);
        return node;
    }
};

// LinearBVHNode Definition
struct alignas(32) LinearBVHNode {
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
};

class BVH {
  public:
    PBRT_GPU explicit BVH(Shape const *const *primitives, int n_primitives) {
        // TODO: rewrite BVH interface
        auto bvh_primitives = new BVHPrimitive[n_primitives];
        for (int idx = 0; idx < n_primitives; ++idx) {
            bvh_primitives[idx] = BVHPrimitive(idx, primitives[idx]->bounds());
        }

        ordered_primitives = new Shape const *[n_primitives];
        int n_ordered_primitives = 0;
        int node_count = 0;

        auto root =
            BVHBuildNode::build_recursive(primitives, bvh_primitives, n_primitives,
                                          ordered_primitives, n_ordered_primitives, node_count);
        delete[] bvh_primitives;

        linear_bvh_nodes = new LinearBVHNode[node_count];

        int array_offset = 0;
        flatten_bvh(root, array_offset);
        delete root;

        printf("BVH built: (%d primitives, %d nodes)\n", n_primitives, node_count);
    }

    PBRT_GPU ~BVH() {
        // TODO: rewrite these array with DynamicArray
        
        delete[] linear_bvh_nodes;
        delete[] ordered_primitives;
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

    // TODO: rewrite these array with DynamicArray
    LinearBVHNode *linear_bvh_nodes = nullptr;
    const Shape **ordered_primitives = nullptr;
};
