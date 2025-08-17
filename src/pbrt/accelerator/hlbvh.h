#pragma once

#include <atomic>
#include <pbrt/base/primitive.h>
#include <pbrt/base/shape.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <vector>

class GPUMemoryAllocator;
class ThreadPool;

class HLBVH {
  public:
    struct MortonPrimitive {
        int primitive_idx;
        uint32_t morton_code;
        Bounds3f bounds;
        Point3f centroid;
    };

    struct Treelet {
        int first_primitive_offset;
        int n_primitives;
        Bounds3f bounds;
    };

    struct BottomBVHArgs {
        int build_node_idx;
        int left_child_idx;
        bool expand_leaf;
    };

    struct BVHBuildNode {
        Bounds3f bounds;

        union {
            int first_primitive_idx; // for leaf node
            int left_child_idx;      // for interior node
        };

        int num_primitives;
        // 0 for interior node
        // otherwise for leaf node

        /*
        uint8_t:         0 - 255
        uint16_t:        0 - 65535
        int (int32_t):   0 - 2147483647
        */

        uint8_t axis;

        PBRT_CPU_GPU
        bool is_leaf() const {
            return num_primitives > 0;
        }

        PBRT_CPU_GPU
        void init_leaf(int _first_primitive_offset, int _num_primitive, const Bounds3f &_bounds) {
            first_primitive_idx = _first_primitive_offset;
            num_primitives = _num_primitive;
            bounds = _bounds;
            axis = 255;
        }

        PBRT_CPU_GPU
        void init_interior(uint8_t _axis, int _left_child_idx, const Bounds3f &_bounds) {
            left_child_idx = _left_child_idx;
            num_primitives = 0;

            bounds = _bounds;
            axis = _axis;
        }
    };

    HLBVH(const std::vector<const Primitive *> &gpu_primitives, const std::string &tag,
          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        if (build_nodes == nullptr) {
            return Bounds3f::empty();
        }

        return build_nodes[0].bounds;
    }

    PBRT_CPU_GPU
    cuda::std::pair<const Primitive **, int> get_primitives() const {
        return {primitives, num_primitives};
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

    PBRT_GPU
    void build_bottom_bvh(const BottomBVHArgs *bvh_args_array, int array_length);

  private:
    int build_top_bvh_for_treelets(const Treelet *treelets, int num_dense_treelets,
                                   ThreadPool &thread_pool);

    void build_upper_sah(int build_node_idx, std::vector<int> treelet_indices,
                         const Treelet *treelets, std::atomic_int &node_count,
                         ThreadPool &thread_pool, bool spawn);

    PBRT_GPU
    int partition_morton_primitives(int start, int end, uint8_t split_dimension, Real split_val);

    const Primitive **primitives = nullptr;
    int num_primitives = 0;

    MortonPrimitive *morton_primitives = nullptr;
    BVHBuildNode *build_nodes = nullptr;
};
