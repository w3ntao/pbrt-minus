#include <chrono>
#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/util/stack.h>
#include <pbrt/util/thread_pool.h>

constexpr uint TREELET_MORTON_BITS_PER_DIMENSION = 10;
constexpr uint BIT_LENGTH_OF_TREELET_MASK = 21;
constexpr uint MASK_OFFSET_BIT = TREELET_MORTON_BITS_PER_DIMENSION * 3 - BIT_LENGTH_OF_TREELET_MASK;

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

[[maybe_unused]] static void _validate_treelet_num() {
    static_assert(BIT_LENGTH_OF_TREELET_MASK > 0 && BIT_LENGTH_OF_TREELET_MASK < 32);
    static_assert(BIT_LENGTH_OF_TREELET_MASK % 3 == 0);
}

constexpr int MORTON_SCALE = 1 << TREELET_MORTON_BITS_PER_DIMENSION;

constexpr uint NUM_BUCKETS = 24;

PBRT_CPU_GPU
uint morton_code_to_treelet_idx(const uint morton_code) {
    const auto masked_morton_code = morton_code & TREELET_MASK;

    return masked_morton_code >>
           (3 * TREELET_MORTON_BITS_PER_DIMENSION - BIT_LENGTH_OF_TREELET_MASK);
}

template <typename T>
__global__ void init_array(T *array, const T val, const uint length) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= length) {
        return;
    }

    array[worker_idx] = val;
}

__global__ void sort_morton_primitives(HLBVH::MortonPrimitive *out,
                                       const HLBVH::MortonPrimitive *in, uint *counter,
                                       const uint *offset, const uint num_primitives) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    const auto primitive = &in[worker_idx];
    const uint treelet_idx = morton_code_to_treelet_idx(primitive->morton_code);

    const uint sorted_idx = atomicAdd(&counter[treelet_idx], 1) + offset[treelet_idx];
    out[sorted_idx] = *primitive;
}

__global__ void count_primitives_for_treelets(uint *counter,
                                              const HLBVH::MortonPrimitive *morton_primitives,
                                              const uint num_primitives) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    const uint morton_code = morton_primitives[worker_idx].morton_code;
    const uint treelet_idx = morton_code_to_treelet_idx(morton_code);
    atomicAdd(&counter[treelet_idx], 1);
}

__global__ void compute_treelet_bounds(HLBVH::Treelet *treelets,
                                       const HLBVH::MortonPrimitive *morton_primitives) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= MAX_TREELET_NUM) {
        return;
    }

    const uint start = treelets[worker_idx].first_primitive_offset;
    const uint end =
        treelets[worker_idx].first_primitive_offset + treelets[worker_idx].n_primitives;

    Bounds3f bounds;
    for (uint primitive_idx = start; primitive_idx < end; ++primitive_idx) {
        bounds += morton_primitives[primitive_idx].bounds;
    }

    treelets[worker_idx].bounds = bounds;
}

__global__ void hlbvh_init_morton_primitives(HLBVH::MortonPrimitive *morton_primitives,
                                             const Primitive **primitives, uint num_primitives) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    morton_primitives[worker_idx].primitive_idx = worker_idx;

    const auto _bounds = primitives[worker_idx]->bounds();

    morton_primitives[worker_idx].bounds = _bounds;
    morton_primitives[worker_idx].centroid = _bounds.centroid();
}

__global__ void hlbvh_init_treelets(HLBVH::Treelet *treelets) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (worker_idx >= MAX_TREELET_NUM) {
        return;
    }

    treelets[worker_idx].first_primitive_offset = std::numeric_limits<uint>::max();
    treelets[worker_idx].n_primitives = 0;
}

__global__ void hlbvh_compute_morton_code(HLBVH::MortonPrimitive *morton_primitives,
                                          uint num_total_primitives,
                                          const Bounds3f bounds_of_centroids) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_total_primitives) {
        return;
    }

    // compute morton code for each primitive
    auto centroid_offset = bounds_of_centroids.offset(morton_primitives[worker_idx].centroid);

    auto scaled_offset = centroid_offset * MORTON_SCALE;
    morton_primitives[worker_idx].morton_code = encode_morton3(
        uint32_t(scaled_offset.x), uint32_t(scaled_offset.y), uint32_t(scaled_offset.z));
}

__global__ void hlbvh_build_bottom_bvh(const HLBVH::BottomBVHArgs *bvh_args_array,
                                       uint array_length, HLBVH *bvh) {
    bvh->build_bottom_bvh(bvh_args_array, array_length);
}

__global__ void init_bvh_args(HLBVH::BottomBVHArgs *bvh_args_array,
                              const HLBVH::BVHBuildNode *bvh_build_nodes, uint *shared_offset,
                              const uint start, const uint end) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint total_jobs = end - start;
    if (worker_idx >= total_jobs) {
        return;
    }

    const uint build_node_idx = worker_idx + start;
    const auto &node = bvh_build_nodes[build_node_idx];

    if (!node.is_leaf() || node.num_primitives <= MAX_PRIMITIVES_NUM_IN_LEAF) {
        bvh_args_array[worker_idx].expand_leaf = false;
        return;
    }

    bvh_args_array[worker_idx].expand_leaf = true;
    bvh_args_array[worker_idx].build_node_idx = build_node_idx;
    bvh_args_array[worker_idx].left_child_idx = atomicAdd(shared_offset, 2);
    // 2 pointers: one for left and another right child
}

const HLBVH *HLBVH::create(const std::vector<const Primitive *> &gpu_primitives,
                           GPUMemoryAllocator &allocator) {
    auto bvh = allocator.allocate<HLBVH>();
    bvh->build_bvh(gpu_primitives, allocator);

    return bvh;
}

PBRT_CPU_GPU
bool HLBVH::fast_intersect(const Ray &ray, FloatType t_max) const {
    if (build_nodes == nullptr) {
        return false;
    }

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
            nodes_to_visit.push(node.left_child_idx + 1);
        } else {
            nodes_to_visit.push(node.left_child_idx + 1);
            nodes_to_visit.push(node.left_child_idx);
        }
    }

    return false;
}

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> HLBVH::intersect(const Ray &ray, FloatType t_max) const {
    if (build_nodes == nullptr) {
        return {};
    }

    pbrt::optional<ShapeIntersection> best_intersection = {};
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
            nodes_to_visit.push(node.left_child_idx + 1);
        } else {
            nodes_to_visit.push(node.left_child_idx + 1);
            nodes_to_visit.push(node.left_child_idx);
        }
    }

    return best_intersection;
};

PBRT_GPU
void HLBVH::build_bottom_bvh(const BottomBVHArgs *bvh_args_array, uint array_length) {
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

    if constexpr (DEBUG_MODE) {
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

        for (uint morton_idx = mid_idx; morton_idx < node.first_primitive_idx + node.num_primitives;
             morton_idx++) {
            if (morton_primitives[morton_idx].centroid[split_dimension] < split_val) {
                printf("ERROR in partitioning (2nd half) at node[%u], idx: %u\n",
                       args.build_node_idx, morton_idx);
                kill_thread = true;
            }
        }

        if (kill_thread) {
            REPORT_FATAL_ERROR();
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
                                                   left_bounds + right_bounds);
}

void HLBVH::build_bvh(const std::vector<const Primitive *> &gpu_primitives,
                      GPUMemoryAllocator &allocator) {
    auto start_sorting = std::chrono::system_clock::now();

    primitives = nullptr;
    morton_primitives = nullptr;
    build_nodes = nullptr;

    uint num_total_primitives = gpu_primitives.size();
    if (num_total_primitives == 0) {
        return;
    }

    printf("\ntotal primitives: %u\n", num_total_primitives);

    GPUMemoryAllocator local_allocator;

    auto sparse_treelets = local_allocator.allocate<Treelet>(MAX_TREELET_NUM);

    auto gpu_morton_primitives = allocator.allocate<MortonPrimitive>(num_total_primitives);
    auto gpu_primitives_array = allocator.allocate<const Primitive *>(num_total_primitives);

    CHECK_CUDA_ERROR(cudaMemcpy(gpu_primitives_array, gpu_primitives.data(),
                                sizeof(Primitive *) * num_total_primitives,
                                cudaMemcpyHostToDevice));

    this->init(gpu_primitives_array, gpu_morton_primitives);

    constexpr uint threads = 1024;
    {
        const uint blocks = divide_and_ceil(num_total_primitives, threads);
        hlbvh_init_morton_primitives<<<blocks, threads>>>(morton_primitives, primitives,
                                                          num_total_primitives);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        const uint blocks = divide_and_ceil(MAX_TREELET_NUM, threads);
        hlbvh_init_treelets<<<blocks, threads>>>(sparse_treelets);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    Bounds3f bounds_of_primitives_centroids;
    for (uint idx = 0; idx < num_total_primitives; idx++) {
        bounds_of_primitives_centroids += gpu_morton_primitives[idx].bounds.centroid();
    }
    auto max_dim = bounds_of_primitives_centroids.max_dimension();
    auto radius = (bounds_of_primitives_centroids.p_max[max_dim] -
                   bounds_of_primitives_centroids.p_min[max_dim]) /
                  2;
    auto adjusted_p_min = bounds_of_primitives_centroids.p_min;
    auto adjusted_p_max = bounds_of_primitives_centroids.p_max;
    for (uint dim = 0; dim < 3; ++dim) {
        if (dim == max_dim) {
            continue;
        }
        auto center = bounds_of_primitives_centroids.centroid()[dim];
        adjusted_p_min[dim] = center - radius;
        adjusted_p_max[dim] = center + radius;
    }
    bounds_of_primitives_centroids = Bounds3f(adjusted_p_min, adjusted_p_max);
    // after adjusting bounds, all treelets grids are of the same size

    {
        const uint blocks = divide_and_ceil(num_total_primitives, threads);
        hlbvh_compute_morton_code<<<blocks, threads>>>(morton_primitives, num_total_primitives,
                                                       bounds_of_primitives_centroids);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto primitives_counter = local_allocator.allocate<uint>(MAX_TREELET_NUM);
    auto primitives_indices_offset = local_allocator.allocate<uint>(MAX_TREELET_NUM);

    {
        const uint blocks = divide_and_ceil(MAX_TREELET_NUM, threads);

        init_array<<<blocks, threads>>>(primitives_counter, uint(0), MAX_TREELET_NUM);
        init_array<<<blocks, threads>>>(primitives_indices_offset, uint(0), MAX_TREELET_NUM);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        const uint blocks = divide_and_ceil(num_total_primitives, threads);
        count_primitives_for_treelets<<<blocks, threads>>>(primitives_counter, morton_primitives,
                                                           num_total_primitives);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    for (uint idx = 1; idx < MAX_TREELET_NUM; ++idx) {
        primitives_indices_offset[idx] =
            primitives_indices_offset[idx - 1] + primitives_counter[idx - 1];
    }

    {
        const uint blocks = divide_and_ceil(MAX_TREELET_NUM, threads);

        init_array<<<blocks, threads>>>(primitives_counter, uint(0), MAX_TREELET_NUM);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto buffer_morton_primitives = local_allocator.allocate<MortonPrimitive>(num_total_primitives);
    {
        const uint blocks = divide_and_ceil(num_total_primitives, threads);
        sort_morton_primitives<<<blocks, threads>>>(buffer_morton_primitives, morton_primitives,
                                                    primitives_counter, primitives_indices_offset,
                                                    num_total_primitives);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    CHECK_CUDA_ERROR(cudaMemcpy(morton_primitives, buffer_morton_primitives,
                                sizeof(MortonPrimitive) * num_total_primitives,
                                cudaMemcpyDeviceToDevice));

    for (uint treelet_idx = 0; treelet_idx < MAX_TREELET_NUM; ++treelet_idx) {
        sparse_treelets[treelet_idx].first_primitive_offset =
            primitives_indices_offset[treelet_idx];
        sparse_treelets[treelet_idx].n_primitives = primitives_counter[treelet_idx];
        // bounds is not computed so far
    }
    {
        // compute bounds
        const uint blocks = divide_and_ceil(MAX_TREELET_NUM, threads);
        compute_treelet_bounds<<<blocks, threads>>>(sparse_treelets, morton_primitives);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    std::vector<uint> dense_treelet_indices;
    {
        uint max_primitive_num_in_a_treelet = 0;
        uint verify_counter = 0;
        for (uint idx = 0; idx < MAX_TREELET_NUM; idx++) {
            uint current_treelet_primitives_num = sparse_treelets[idx].n_primitives;
            if (current_treelet_primitives_num <= 0) {
                continue;
            }

            verify_counter += current_treelet_primitives_num;

            max_primitive_num_in_a_treelet =
                std::max(max_primitive_num_in_a_treelet, current_treelet_primitives_num);
            dense_treelet_indices.push_back(idx);
        }

        if (verify_counter != num_total_primitives) {
            REPORT_FATAL_ERROR();
        }

        printf("HLBVH: %zu/%d (%.2f%) treelets filled (max primitives in a treelet: %d)\n",
               dense_treelet_indices.size(), MAX_TREELET_NUM,
               double(dense_treelet_indices.size()) / MAX_TREELET_NUM * 100,
               max_primitive_num_in_a_treelet);
    }

    auto dense_treelets = local_allocator.allocate<Treelet>(dense_treelet_indices.size());
    for (uint idx = 0; idx < dense_treelet_indices.size(); idx++) {
        uint sparse_idx = dense_treelet_indices[idx];
        CHECK_CUDA_ERROR(cudaMemcpy(&dense_treelets[idx], &sparse_treelets[sparse_idx],
                                    sizeof(Treelet), cudaMemcpyDeviceToDevice));
    }

    uint max_build_node_length =
        (2 * dense_treelet_indices.size() + 1) + (2 * num_total_primitives + 1);

    build_nodes = allocator.allocate<BVHBuildNode>(max_build_node_length);

    auto start_top_bvh = std::chrono::system_clock::now();

    ThreadPool thread_pool;
    const uint top_bvh_node_num =
        build_top_bvh_for_treelets(dense_treelets, dense_treelet_indices.size(), thread_pool);

    auto start_bottom_bvh = std::chrono::system_clock::now();

    uint start = 0;
    uint end = top_bvh_node_num;

    auto shared_offset = local_allocator.allocate<uint>();
    *shared_offset = end;

    uint depth = 0;

    uint last_allocated_size = (end - start) * 4;
    auto bvh_args_array = local_allocator.allocate<BottomBVHArgs>(last_allocated_size);

    while (end > start) {
        const uint array_length = end - start;

        if (array_length > last_allocated_size) {
            // to avoid unnecessarily repeated memory allocation
            uint current_size = array_length * 2;
            bvh_args_array = local_allocator.allocate<BottomBVHArgs>(current_size);
            last_allocated_size = current_size;
        }

        {
            uint blocks = divide_and_ceil(array_length, threads);
            init_bvh_args<<<blocks, threads>>>(bvh_args_array, build_nodes, shared_offset, start,
                                               end);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        if constexpr (DEBUG_MODE) {
            printf("HLBVH: building bottom BVH: depth %u, node number: %u\n", depth, array_length);
        }

        depth += 1;
        start = end;
        end = *shared_offset;

        uint blocks = divide_and_ceil(array_length, threads);

        hlbvh_build_bottom_bvh<<<blocks, threads>>>(bvh_args_array, array_length, this);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    printf("HLBVH: bottom BVH nodes: %u, max depth: %u, max primitives in a leaf: %u\n",
           end - top_bvh_node_num, depth, MAX_PRIMITIVES_NUM_IN_LEAF);
    printf("HLBVH: total nodes: %u/%u\n", end, max_build_node_length);

    const std::chrono::duration<FloatType> duration_sorting{start_top_bvh - start_sorting};

    const std::chrono::duration<FloatType> duration_top_bvh{start_bottom_bvh - start_sorting};

    const std::chrono::duration<FloatType> duration_bottom_bvh{std::chrono::system_clock::now() -
                                                               start_bottom_bvh};

    printf("BVH constructing took %.2f seconds "
           "(sorting: %.2f, top SAH-BVH building: %.2f, bottom BVH building: %.2f)\n",
           (duration_sorting + duration_top_bvh + duration_bottom_bvh).count(),
           duration_sorting.count(), duration_top_bvh.count(), duration_bottom_bvh.count());
}

uint HLBVH::build_top_bvh_for_treelets(const Treelet *treelets, const uint num_dense_treelets,
                                       ThreadPool &thread_pool) {
    std::vector<uint> treelet_indices;
    treelet_indices.reserve(num_dense_treelets);
    for (uint idx = 0; idx < num_dense_treelets; idx++) {
        treelet_indices.emplace_back(idx);
    }

    std::atomic_int node_count = 1; // the first index used for the root

    thread_pool.submit([this, _treelet_indices = std::move(treelet_indices), &treelets, &node_count,
                        &thread_pool] {
        build_upper_sah(0, _treelet_indices, treelets, std::ref(node_count), std::ref(thread_pool),
                        true);
    });
    thread_pool.sync();

    printf("HLBVH: build top BVH with SAH using %d buckets\n", NUM_BUCKETS);
    printf("HLBVH: top BVH nodes: %u\n", node_count.load());

    return node_count.load();
}

void HLBVH::build_upper_sah(uint build_node_idx, std::vector<uint> treelet_indices,
                            const Treelet *treelets, std::atomic_int &node_count,
                            ThreadPool &thread_pool, bool spawn) {
    if (treelet_indices.size() == 1) {
        uint treelet_idx = treelet_indices[0];
        const auto &current_treelet = treelets[treelet_idx];

        build_nodes[build_node_idx].init_leaf(current_treelet.first_primitive_offset,
                                              current_treelet.n_primitives, current_treelet.bounds);
        return;
    }

    Bounds3f full_bounds_of_current_level;
    Bounds3f bounds_of_centroid;
    for (const auto treelet_idx : treelet_indices) {
        bounds_of_centroid += treelets[treelet_idx].bounds.centroid();
        full_bounds_of_current_level += treelets[treelet_idx].bounds;
    }

    const uint8_t split_axis = bounds_of_centroid.max_dimension();

    if (bounds_of_centroid.p_min[split_axis] == bounds_of_centroid.p_max[split_axis]) {
        // when the bounds is of zero volume
        // should build everything into one leaf?
        REPORT_FATAL_ERROR();
    }

    struct BVHSplitBucket {
        uint count = 0;
        Bounds3f bounds = Bounds3f ::empty();
        std::vector<uint> treelet_indices;
    };
    BVHSplitBucket buckets[NUM_BUCKETS];
    for (auto &bucket : buckets) {
        bucket.treelet_indices.reserve(treelet_indices.size() / NUM_BUCKETS * 2);
    }

    const auto base_val = bounds_of_centroid.p_min[split_axis];
    const auto span = bounds_of_centroid.p_max[split_axis] - bounds_of_centroid.p_min[split_axis];

    // Initialize _BVHSplitBucket_ for HLBVH SAH partition buckets
    for (unsigned int treelet_idx : treelet_indices) {
        const auto treelet = &treelets[treelet_idx];

        auto centroid_val = treelet->bounds.centroid()[split_axis];
        uint bucket_idx = NUM_BUCKETS * ((centroid_val - base_val) / span);

        if (bucket_idx > NUM_BUCKETS) {
            REPORT_FATAL_ERROR();
        }
        if (bucket_idx == NUM_BUCKETS) {
            bucket_idx = NUM_BUCKETS - 1;
        }

        buckets[bucket_idx].count += treelet->n_primitives;
        buckets[bucket_idx].bounds += treelet->bounds;
        buckets[bucket_idx].treelet_indices.emplace_back(treelet_idx);
    }

    const auto total_surface_area = full_bounds_of_current_level.surface_area();

    // Compute costs for splitting after each bucket
    FloatType sah_cost[NUM_BUCKETS - 1];
    for (uint split_idx = 0; split_idx < NUM_BUCKETS - 1; ++split_idx) {
        Bounds3f bounds_left;
        Bounds3f bounds_right;
        uint count_left = 0;
        uint count_right = 0;

        for (uint left = 0; left <= split_idx; ++left) {
            bounds_left += buckets[left].bounds;
            count_left += buckets[left].count;
        }

        for (uint right = split_idx + 1; right < NUM_BUCKETS; ++right) {
            bounds_right += buckets[right].bounds;
            count_right += buckets[right].count;
        }

        sah_cost[split_idx] = 0.125 + (count_left * bounds_left.surface_area() +
                                       count_right * bounds_right.surface_area()) /
                                          total_surface_area;
    }

    // Find bucket to split at that minimizes SAH metric
    FloatType min_cost_so_far = sah_cost[0];
    uint min_cost_split = 0;
    for (uint idx = 1; idx < NUM_BUCKETS - 1; ++idx) {
        if (sah_cost[idx] < min_cost_so_far) {
            min_cost_so_far = sah_cost[idx];
            min_cost_split = idx;
        }
    }

    std::vector<uint> left_indices;
    std::vector<uint> right_indices;
    left_indices.reserve(treelet_indices.size());
    right_indices.reserve(treelet_indices.size());

    for (uint idx = 0; idx <= min_cost_split; ++idx) {
        left_indices.insert(left_indices.end(), buckets[idx].treelet_indices.begin(),
                            buckets[idx].treelet_indices.end());
    }

    for (uint idx = min_cost_split + 1; idx < NUM_BUCKETS; ++idx) {
        right_indices.insert(right_indices.end(), buckets[idx].treelet_indices.begin(),
                             buckets[idx].treelet_indices.end());
    }

    if (left_indices.empty() || right_indices.empty()) {
        // when SAH couldn't build a valid tree: fall back to MidSplit
        left_indices.clear();
        right_indices.clear();

        auto split_val = bounds_of_centroid.centroid()[split_axis];
        for (const auto idx : treelet_indices) {
            if (treelets[idx].bounds.centroid()[split_axis] < split_val) {
                left_indices.emplace_back(idx);
            } else {
                right_indices.emplace_back(idx);
            }
        }

        if (left_indices.empty() || right_indices.empty()) {
            // if MidSplit still couldn't divide them:
            // should build everything into one leaf?
            REPORT_FATAL_ERROR();
        }
    }

    if constexpr (DEBUG_MODE) {
        // check missing indices
        std::vector<uint> combined_indices = left_indices;
        combined_indices.insert(combined_indices.end(), right_indices.begin(), right_indices.end());

        auto treelet_indices_copy = treelet_indices;
        std::sort(treelet_indices_copy.begin(), treelet_indices_copy.end());
        std::sort(combined_indices.begin(), combined_indices.end());

        if (treelet_indices_copy != combined_indices) {
            printf("%s(): SAH-BVH not split right\n", __func__);
            REPORT_FATAL_ERROR();
        }
    }

    treelet_indices.clear();

    const uint left_build_node_idx = node_count.fetch_add(2);
    const uint right_build_node_idx = left_build_node_idx + 1;

    build_nodes[build_node_idx].init_interior(split_axis, left_build_node_idx,
                                              full_bounds_of_current_level);

    constexpr uint MIN_SIZE_TO_SPAWN = 64;
    // don't bother to send jobs into queue when the size is too small

    if (spawn && left_indices.size() >= MIN_SIZE_TO_SPAWN) {
        thread_pool.submit([this, left_build_node_idx, _left_indices = std::move(left_indices),
                            &treelets, &node_count, &thread_pool] {
            build_upper_sah(left_build_node_idx, _left_indices, treelets, std::ref(node_count),
                            std::ref(thread_pool), true);
        });
    } else {
        build_upper_sah(left_build_node_idx, std::move(left_indices), treelets,
                        std::ref(node_count), std::ref(thread_pool), false);
    }

    if (spawn && right_indices.size() >= MIN_SIZE_TO_SPAWN) {
        thread_pool.submit([this, right_build_node_idx, _right_indices = std::move(right_indices),
                            &treelets, &node_count, &thread_pool] {
            build_upper_sah(right_build_node_idx, _right_indices, treelets, std::ref(node_count),
                            std::ref(thread_pool), true);
        });
    } else {
        build_upper_sah(right_build_node_idx, std::move(right_indices), treelets,
                        std::ref(node_count), std::ref(thread_pool), false);
    }
}

PBRT_GPU
uint HLBVH::partition_morton_primitives(const uint start, const uint end,
                                        const uint8_t split_dimension, const FloatType split_val) {
    // taken and modified from
    // https://users.cs.duke.edu/~reif/courses/alglectures/littman.lectures/lect05/node27.html

    uint left = start;
    uint right = end - 1;

    while (true) {
        while (morton_primitives[right].centroid[split_dimension] >= split_val && right > start) {
            right--;
        }

        while (morton_primitives[left].centroid[split_dimension] < split_val && left < end - 1) {
            left++;
        }

        if (left < right) {
            pbrt::swap(morton_primitives[left], morton_primitives[right]);
            continue;
        }

        if (left == start && right == start) {
            return start;
        }

        return right + 1;
    }
}
