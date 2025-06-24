#include <chrono>
#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/util/stack.h>
#include <pbrt/util/thread_pool.h>

constexpr int STACK_SIZE = 128;

constexpr int TREELET_MORTON_BITS_PER_DIMENSION = 10;
constexpr int BIT_LENGTH_OF_TREELET_MASK = 21;
constexpr int MASK_OFFSET_BIT = TREELET_MORTON_BITS_PER_DIMENSION * 3 - BIT_LENGTH_OF_TREELET_MASK;

constexpr int MAX_TREELET_NUM = 1 << BIT_LENGTH_OF_TREELET_MASK;
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

constexpr int MAX_PRIMITIVES_NUM_IN_LEAF = 1;

[[maybe_unused]] static void _validate_treelet_num() {
    static_assert(BIT_LENGTH_OF_TREELET_MASK > 0 && BIT_LENGTH_OF_TREELET_MASK < 32);
    static_assert(BIT_LENGTH_OF_TREELET_MASK % 3 == 0);
}

constexpr int MORTON_SCALE = 1 << TREELET_MORTON_BITS_PER_DIMENSION;

constexpr int NUM_BUCKETS = 24;

PBRT_CPU_GPU
int morton_code_to_treelet_idx(const int morton_code) {
    const auto masked_morton_code = morton_code & TREELET_MASK;

    return masked_morton_code >>
           (3 * TREELET_MORTON_BITS_PER_DIMENSION - BIT_LENGTH_OF_TREELET_MASK);
}

template <typename T>
__global__ void init_array(T *array, const T val, const int length) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= length) {
        return;
    }

    array[worker_idx] = val;
}

__global__ void sort_morton_primitives(HLBVH::MortonPrimitive *out,
                                       const HLBVH::MortonPrimitive *in, int *counter,
                                       const int *offset, const int num_primitives) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    const auto primitive = &in[worker_idx];
    const int treelet_idx = morton_code_to_treelet_idx(primitive->morton_code);

    const int sorted_idx = atomicAdd(&counter[treelet_idx], 1) + offset[treelet_idx];
    out[sorted_idx] = *primitive;
}

__global__ void count_primitives_for_treelets(int *counter,
                                              const HLBVH::MortonPrimitive *morton_primitives,
                                              const int num_primitives) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    const int morton_code = morton_primitives[worker_idx].morton_code;
    const int treelet_idx = morton_code_to_treelet_idx(morton_code);
    atomicAdd(&counter[treelet_idx], 1);
}

__global__ void compute_treelet_bounds(HLBVH::Treelet *treelets,
                                       const HLBVH::MortonPrimitive *morton_primitives) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= MAX_TREELET_NUM) {
        return;
    }

    const int start = treelets[worker_idx].first_primitive_offset;
    const int end = treelets[worker_idx].first_primitive_offset + treelets[worker_idx].n_primitives;

    Bounds3f bounds;
    for (int primitive_idx = start; primitive_idx < end; ++primitive_idx) {
        bounds += morton_primitives[primitive_idx].bounds;
    }

    treelets[worker_idx].bounds = bounds;
}

__global__ void hlbvh_init_morton_primitives(HLBVH::MortonPrimitive *morton_primitives,
                                             const Primitive **primitives, int num_primitives) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    morton_primitives[worker_idx].primitive_idx = worker_idx;

    const auto _bounds = primitives[worker_idx]->bounds();

    morton_primitives[worker_idx].bounds = _bounds;
    morton_primitives[worker_idx].centroid = _bounds.centroid();
}

__global__ void hlbvh_init_treelets(HLBVH::Treelet *treelets) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (worker_idx >= MAX_TREELET_NUM) {
        return;
    }

    treelets[worker_idx].first_primitive_offset = std::numeric_limits<int>::max();
    treelets[worker_idx].n_primitives = 0;
}

__global__ void hlbvh_compute_morton_code(HLBVH::MortonPrimitive *morton_primitives,
                                          int num_total_primitives,
                                          const Bounds3f bounds_of_centroids) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_total_primitives) {
        return;
    }

    // compute morton code for each primitive
    auto centroid_offset = bounds_of_centroids.offset(morton_primitives[worker_idx].centroid);

    auto scaled_offset = centroid_offset * MORTON_SCALE;
    morton_primitives[worker_idx].morton_code = encode_morton3(
        uint32_t(scaled_offset.x), uint32_t(scaled_offset.y), uint32_t(scaled_offset.z));
}

__global__ void hlbvh_build_bottom_bvh(const HLBVH::BottomBVHArgs *bvh_args_array, int array_length,
                                       HLBVH *bvh) {
    bvh->build_bottom_bvh(bvh_args_array, array_length);
}

__global__ void init_bvh_args(HLBVH::BottomBVHArgs *bvh_args_array,
                              const HLBVH::BVHBuildNode *bvh_build_nodes, int *shared_offset,
                              const int start, const int end) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_jobs = end - start;
    if (worker_idx >= total_jobs) {
        return;
    }

    const int build_node_idx = worker_idx + start;
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
                           const std::string &tag, GPUMemoryAllocator &allocator) {
    auto bvh = allocator.allocate<HLBVH>();
    bvh->build_bvh(gpu_primitives, tag, allocator);

    return bvh;
}

PBRT_CPU_GPU
bool HLBVH::fast_intersect(const Ray &ray, Real t_max) const {
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

    Stack<int, STACK_SIZE> nodes_to_visit;
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
            for (int morton_idx = node.first_primitive_idx;
                 morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
                const int primitive_idx = morton_primitives[morton_idx].primitive_idx;
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
pbrt::optional<ShapeIntersection> HLBVH::intersect(const Ray &ray, Real t_max) const {
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

    Stack<int, STACK_SIZE> nodes_to_visit;
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
            for (int morton_idx = node.first_primitive_idx;
                 morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
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
void HLBVH::build_bottom_bvh(const BottomBVHArgs *bvh_args_array, int array_length) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= array_length) {
        return;
    }

    const auto &args = bvh_args_array[worker_idx];

    if (!args.expand_leaf) {
        return;
    }

    const auto &node = build_nodes[args.build_node_idx];
    int left_child_idx = args.left_child_idx + 0;
    int right_child_idx = args.left_child_idx + 1;

    Bounds3f bounds_of_centroid;
    for (int morton_idx = node.first_primitive_idx;
         morton_idx < node.first_primitive_idx + node.num_primitives; morton_idx++) {
        bounds_of_centroid += morton_primitives[morton_idx].centroid;
    }

    auto split_dimension = bounds_of_centroid.max_dimension();
    auto split_val = bounds_of_centroid.centroid()[split_dimension];

    int mid_idx = partition_morton_primitives(node.first_primitive_idx,
                                              node.first_primitive_idx + node.num_primitives,
                                              split_dimension, split_val);

    if (DEBUG_MODE) {
        bool kill_thread = false;

        if (mid_idx < node.first_primitive_idx ||
            mid_idx > node.first_primitive_idx + node.num_primitives) {
            printf("ERROR in partitioning at node[%u]: mid_idx out of bound\n",
                   args.build_node_idx);
            kill_thread = true;
        }

        for (int morton_idx = node.first_primitive_idx; morton_idx < mid_idx; morton_idx++) {
            if (morton_primitives[morton_idx].centroid[split_dimension] >= split_val) {
                printf("ERROR in partitioning (1st half) at node[%u], idx: %u\n",
                       args.build_node_idx, morton_idx);
                kill_thread = true;
            }
        }

        for (int morton_idx = mid_idx; morton_idx < node.first_primitive_idx + node.num_primitives;
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
    for (int morton_idx = node.first_primitive_idx; morton_idx < mid_idx; morton_idx++) {
        left_bounds += morton_primitives[morton_idx].bounds;
    }

    Bounds3f right_bounds;
    for (int morton_idx = mid_idx; morton_idx < node.first_primitive_idx + node.num_primitives;
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

void HLBVH::build_bvh(const std::vector<const Primitive *> &gpu_primitives, const std::string &tag,
                      GPUMemoryAllocator &allocator) {
    auto start_sorting = std::chrono::system_clock::now();

    primitives = nullptr;
    morton_primitives = nullptr;
    build_nodes = nullptr;

    num_primitives = gpu_primitives.size();
    if (num_primitives == 0) {
        return;
    }

    GPUMemoryAllocator local_allocator;

    auto sparse_treelets = local_allocator.allocate<Treelet>(MAX_TREELET_NUM);

    morton_primitives = allocator.allocate<MortonPrimitive>(num_primitives);
    primitives = allocator.allocate<const Primitive *>(num_primitives);

    CHECK_CUDA_ERROR(cudaMemcpy(primitives, gpu_primitives.data(),
                                sizeof(Primitive *) * num_primitives, cudaMemcpyHostToDevice));

    constexpr int threads = 1024;
    {
        const int blocks = divide_and_ceil(num_primitives, threads);
        hlbvh_init_morton_primitives<<<blocks, threads>>>(morton_primitives, primitives,
                                                          num_primitives);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        const int blocks = divide_and_ceil(MAX_TREELET_NUM, threads);
        hlbvh_init_treelets<<<blocks, threads>>>(sparse_treelets);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    Bounds3f bounds_of_primitives_centroids;
    for (int idx = 0; idx < num_primitives; idx++) {
        bounds_of_primitives_centroids += morton_primitives[idx].bounds.centroid();
    }
    auto max_dim = bounds_of_primitives_centroids.max_dimension();
    auto radius = (bounds_of_primitives_centroids.p_max[max_dim] -
                   bounds_of_primitives_centroids.p_min[max_dim]) /
                  2;
    auto adjusted_p_min = bounds_of_primitives_centroids.p_min;
    auto adjusted_p_max = bounds_of_primitives_centroids.p_max;
    for (int dim = 0; dim < 3; ++dim) {
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
        const int blocks = divide_and_ceil(num_primitives, threads);
        hlbvh_compute_morton_code<<<blocks, threads>>>(morton_primitives, num_primitives,
                                                       bounds_of_primitives_centroids);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto primitives_counter = local_allocator.allocate<int>(MAX_TREELET_NUM);
    auto primitives_indices_offset = local_allocator.allocate<int>(MAX_TREELET_NUM);

    {
        const int blocks = divide_and_ceil(MAX_TREELET_NUM, threads);

        init_array<<<blocks, threads>>>(primitives_counter, int(0), MAX_TREELET_NUM);
        init_array<<<blocks, threads>>>(primitives_indices_offset, int(0), MAX_TREELET_NUM);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    {
        const int blocks = divide_and_ceil(num_primitives, threads);
        count_primitives_for_treelets<<<blocks, threads>>>(primitives_counter, morton_primitives,
                                                           num_primitives);

        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    for (int idx = 1; idx < MAX_TREELET_NUM; ++idx) {
        primitives_indices_offset[idx] =
            primitives_indices_offset[idx - 1] + primitives_counter[idx - 1];
    }

    {
        const int blocks = divide_and_ceil(MAX_TREELET_NUM, threads);

        init_array<<<blocks, threads>>>(primitives_counter, int(0), MAX_TREELET_NUM);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto buffer_morton_primitives = local_allocator.allocate<MortonPrimitive>(num_primitives);
    {
        const int blocks = divide_and_ceil(num_primitives, threads);
        sort_morton_primitives<<<blocks, threads>>>(buffer_morton_primitives, morton_primitives,
                                                    primitives_counter, primitives_indices_offset,
                                                    num_primitives);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    CHECK_CUDA_ERROR(cudaMemcpy(morton_primitives, buffer_morton_primitives,
                                sizeof(MortonPrimitive) * num_primitives,
                                cudaMemcpyDeviceToDevice));

    for (int treelet_idx = 0; treelet_idx < MAX_TREELET_NUM; ++treelet_idx) {
        sparse_treelets[treelet_idx].first_primitive_offset =
            primitives_indices_offset[treelet_idx];
        sparse_treelets[treelet_idx].n_primitives = primitives_counter[treelet_idx];
        // bounds is not computed so far
    }
    {
        // compute bounds
        const int blocks = divide_and_ceil(MAX_TREELET_NUM, threads);
        compute_treelet_bounds<<<blocks, threads>>>(sparse_treelets, morton_primitives);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    std::vector<int> dense_treelet_indices;
    {
        int max_primitive_num_in_a_treelet = 0;
        int verify_counter = 0;
        for (int idx = 0; idx < MAX_TREELET_NUM; idx++) {
            int current_treelet_primitives_num = sparse_treelets[idx].n_primitives;
            if (current_treelet_primitives_num <= 0) {
                continue;
            }

            verify_counter += current_treelet_primitives_num;

            max_primitive_num_in_a_treelet =
                std::max(max_primitive_num_in_a_treelet, current_treelet_primitives_num);
            dense_treelet_indices.push_back(idx);
        }

        if (verify_counter != num_primitives) {
            REPORT_FATAL_ERROR();
        }

        if (DEBUG_MODE) {
            printf("HLBVH: %zu/%d (%.2f%) treelets filled (max primitives in a treelet: %d)\n",
                   dense_treelet_indices.size(), MAX_TREELET_NUM,
                   double(dense_treelet_indices.size()) / MAX_TREELET_NUM * 100,
                   max_primitive_num_in_a_treelet);
        }
    }

    auto dense_treelets = local_allocator.allocate<Treelet>(dense_treelet_indices.size());
    for (int idx = 0; idx < dense_treelet_indices.size(); idx++) {
        const int sparse_idx = dense_treelet_indices[idx];
        dense_treelets[idx] = sparse_treelets[sparse_idx];
    }

    int max_build_node_length = (2 * dense_treelet_indices.size() + 1) + (2 * num_primitives + 1);

    build_nodes = allocator.allocate<BVHBuildNode>(max_build_node_length);

    auto start_top_bvh = std::chrono::system_clock::now();

    ThreadPool thread_pool;
    // TODO: increase thread_num in ThreadPool when building top BVH
    const int top_bvh_node_num =
        build_top_bvh_for_treelets(dense_treelets, dense_treelet_indices.size(), thread_pool);

    auto start_bottom_bvh = std::chrono::system_clock::now();

    int start = 0;
    int end = top_bvh_node_num;

    auto shared_offset = local_allocator.allocate<int>();
    *shared_offset = end;

    int depth = 0;

    int last_allocated_size = (end - start) * 4;
    auto bvh_args_array = local_allocator.allocate<BottomBVHArgs>(last_allocated_size);

    while (end > start) {
        const int array_length = end - start;

        if (array_length > last_allocated_size) {
            // to avoid unnecessarily repeated memory allocation
            int current_size = array_length;
            bvh_args_array = local_allocator.allocate<BottomBVHArgs>(current_size);
            last_allocated_size = current_size;
        }

        {
            int blocks = divide_and_ceil(array_length, threads);
            init_bvh_args<<<blocks, threads>>>(bvh_args_array, build_nodes, shared_offset, start,
                                               end);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        if (DEBUG_MODE) {
            printf("HLBVH: building bottom BVH: depth %u, node number: %u\n", depth, array_length);
        }

        depth += 1;
        start = end;
        end = *shared_offset;

        int blocks = divide_and_ceil(array_length, threads);

        hlbvh_build_bottom_bvh<<<blocks, threads>>>(bvh_args_array, array_length, this);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    if (DEBUG_MODE) {
        printf("HLBVH: bottom BVH nodes: %u, max depth: %u, max primitives in a leaf: %u\n",
               end - top_bvh_node_num, depth, MAX_PRIMITIVES_NUM_IN_LEAF);
    }

    const std::chrono::duration<Real> duration_sorting{start_top_bvh - start_sorting};

    const std::chrono::duration<Real> duration_top_bvh{start_bottom_bvh - start_sorting};

    const std::chrono::duration<Real> duration_bottom_bvh{std::chrono::system_clock::now() -
                                                          start_bottom_bvh};

    printf("BVH building: %d primitives, %.2f seconds (sort: %.2f, top: %.2f, bottom: %.2f) %s\n",
           num_primitives, (duration_sorting + duration_top_bvh + duration_bottom_bvh).count(),
           duration_sorting.count(), duration_top_bvh.count(), duration_bottom_bvh.count(),
           tag.c_str());
}

int HLBVH::build_top_bvh_for_treelets(const Treelet *treelets, const int num_dense_treelets,
                                      ThreadPool &thread_pool) {
    std::vector<int> treelet_indices;
    treelet_indices.reserve(num_dense_treelets);
    for (int idx = 0; idx < num_dense_treelets; idx++) {
        treelet_indices.emplace_back(idx);
    }

    std::atomic_int node_count = 1; // the first index used for the root

    thread_pool.submit([this, _treelet_indices = std::move(treelet_indices), &treelets, &node_count,
                        &thread_pool] {
        build_upper_sah(0, _treelet_indices, treelets, std::ref(node_count), std::ref(thread_pool),
                        true);
    });
    thread_pool.sync();

    /*
    printf("HLBVH: build top BVH with SAH using %d buckets\n", NUM_BUCKETS);
    printf("HLBVH: top BVH nodes: %u\n", node_count.load());
    */

    return node_count.load();
}

void HLBVH::build_upper_sah(int build_node_idx, std::vector<int> treelet_indices,
                            const Treelet *treelets, std::atomic_int &node_count,
                            ThreadPool &thread_pool, bool spawn) {
    if (treelet_indices.size() == 1) {
        int treelet_idx = treelet_indices[0];
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
        int count = 0;
        Bounds3f bounds = Bounds3f ::empty();
        std::vector<int> treelet_indices;
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
        int bucket_idx = NUM_BUCKETS * ((centroid_val - base_val) / span);

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
    Real sah_cost[NUM_BUCKETS - 1];
    for (int split_idx = 0; split_idx < NUM_BUCKETS - 1; ++split_idx) {
        Bounds3f bounds_left;
        Bounds3f bounds_right;
        int count_left = 0;
        int count_right = 0;

        for (int left = 0; left <= split_idx; ++left) {
            bounds_left += buckets[left].bounds;
            count_left += buckets[left].count;
        }

        for (int right = split_idx + 1; right < NUM_BUCKETS; ++right) {
            bounds_right += buckets[right].bounds;
            count_right += buckets[right].count;
        }

        sah_cost[split_idx] = 0.125 + (count_left * bounds_left.surface_area() +
                                       count_right * bounds_right.surface_area()) /
                                          total_surface_area;
    }

    // Find bucket to split at that minimizes SAH metric
    Real min_cost_so_far = sah_cost[0];
    int min_cost_split = 0;
    for (int idx = 1; idx < NUM_BUCKETS - 1; ++idx) {
        if (sah_cost[idx] < min_cost_so_far) {
            min_cost_so_far = sah_cost[idx];
            min_cost_split = idx;
        }
    }

    std::vector<int> left_indices;
    std::vector<int> right_indices;
    left_indices.reserve(treelet_indices.size());
    right_indices.reserve(treelet_indices.size());

    for (int idx = 0; idx <= min_cost_split; ++idx) {
        left_indices.insert(left_indices.end(), buckets[idx].treelet_indices.begin(),
                            buckets[idx].treelet_indices.end());
    }

    for (int idx = min_cost_split + 1; idx < NUM_BUCKETS; ++idx) {
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

    if (DEBUG_MODE) {
        // check missing indices
        std::vector<int> combined_indices = left_indices;
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

    const int left_build_node_idx = node_count.fetch_add(2);
    const int right_build_node_idx = left_build_node_idx + 1;

    build_nodes[build_node_idx].init_interior(split_axis, left_build_node_idx,
                                              full_bounds_of_current_level);

    constexpr int MIN_SIZE_TO_SPAWN = 64;
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
int HLBVH::partition_morton_primitives(const int start, const int end,
                                       const uint8_t split_dimension, const Real split_val) {
    // taken and modified from
    // https://users.cs.duke.edu/~reif/courses/alglectures/littman.lectures/lect05/node27.html

    int left = start;
    int right = end - 1;

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
