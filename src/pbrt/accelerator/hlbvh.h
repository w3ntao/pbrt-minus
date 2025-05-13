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
        bool is_leaf() const {
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

    static const HLBVH *create(const std::vector<const Primitive *> &gpu_primitives,
                               const std::string &tag, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        if (build_nodes == nullptr) {
            return Bounds3f::empty();
        }

        return build_nodes[0].bounds;
    }

    PBRT_CPU_GPU
    cuda::std::pair<const Primitive **, uint> get_primitives() const {
        return {primitives, num_primtives};
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

    PBRT_GPU
    void build_bottom_bvh(const BottomBVHArgs *bvh_args_array, uint array_length);

  private:
    void build_bvh(const std::vector<const Primitive *> &gpu_primitives, const std::string &tag,
                   GPUMemoryAllocator &allocator);

    uint build_top_bvh_for_treelets(const Treelet *treelets, uint num_dense_treelets,
                                    ThreadPool &thread_pool);

    void build_upper_sah(uint build_node_idx, std::vector<uint> treelet_indices,
                         const Treelet *treelets, std::atomic_int &node_count,
                         ThreadPool &thread_pool, bool spawn);

    PBRT_GPU
    uint partition_morton_primitives(uint start, uint end, uint8_t split_dimension,
                                     Real split_val);

    const Primitive **primitives;
    uint num_primtives;

    MortonPrimitive *morton_primitives;
    BVHBuildNode *build_nodes;
};
