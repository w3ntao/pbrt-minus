#include "pbrt/accelerator/hlbvh.h"

#include <chrono>
#include <iomanip>
#include <cuda/atomic>

#include "pbrt/util/stack.h"

__global__ void hlbvh_init_morton_primitives(MortonPrimitive *morton_primitives,
                                             const Shape **primitives, uint num_primitives) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_primitives) {
        return;
    }

    morton_primitives[worker_idx].primitive_idx = worker_idx;

    const auto _bounds = primitives[worker_idx]->bounds();

    morton_primitives[worker_idx].bounds = _bounds;
    morton_primitives[worker_idx].centroid = _bounds.centroid();
}

__global__ void hlbvh_init_treelets(Treelet *treelets) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (worker_idx >= MAX_TREELET_NUM) {
        return;
    }

    treelets[worker_idx].first_primitive_offset = std::numeric_limits<uint>::max();
    treelets[worker_idx].n_primitives = 0;
}

__global__ void hlbvh_compute_morton_code(MortonPrimitive *morton_primitives,
                                          uint num_total_primitives,
                                          const Bounds3f bounds_of_centroids) {
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

__global__ void hlbvh_collect_primitives_into_treelets(Treelet *treelets,
                                                       const MortonPrimitive *morton_primitives,
                                                       const Shape **primitives,
                                                       uint num_total_primitives) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_total_primitives) {
        return;
    }

    const uint start = worker_idx;
    uint32_t morton_start = morton_primitives[start].morton_code & TREELET_MASK;

    if (start == 0 || morton_start != (morton_primitives[start - 1].morton_code & TREELET_MASK)) {
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

__global__ void hlbvh_build_bottom_bvh(HLBVH *bvh, const BottomBVHArgs *bvh_args_array,
                                       uint array_length) {
    bvh->build_bottom_bvh(bvh_args_array, array_length);
}

__global__ void init_bvh_args(BottomBVHArgs *bvh_args_array, uint *accumulated_offset,
                              const BVHBuildNode *bvh_build_nodes, const uint start,
                              const uint end) {
    if (gridDim.x * gridDim.y * gridDim.z > 1) {
        printf("init_bvh_args(): launching more than 1 blocks destroys inter-thread "
               "synchronization.\n");
        asm("trap;");
    }

    const uint worker_idx = threadIdx.x;

    __shared__ cuda::atomic<uint, cuda::thread_scope_block> shared_accumulated_offset;
    if (worker_idx == 0) {
        shared_accumulated_offset = *accumulated_offset;
    }
    __syncthreads();

    const uint total_jobs = end - start;
    const uint jobs_per_worker = total_jobs / blockDim.x + 1;

    for (uint job_offset = 0; job_offset < jobs_per_worker; job_offset++) {
        const uint idx = worker_idx * jobs_per_worker + job_offset;
        if (idx >= total_jobs) {
            break;
        }

        const uint build_node_idx = idx + start;
        const auto &node = bvh_build_nodes[build_node_idx];

        if (!node.is_leaf() || node.num_primitives <= MAX_PRIMITIVES_NUM_IN_LEAF) {
            bvh_args_array[idx].expand_leaf = false;
            continue;
        }

        bvh_args_array[idx].expand_leaf = true;
        bvh_args_array[idx].build_node_idx = build_node_idx;
        bvh_args_array[idx].left_child_idx = shared_accumulated_offset.fetch_add(2);
        // 2 pointers: one for left and another right child
    }

    __syncthreads();
    if (worker_idx == 0) {
        *accumulated_offset = shared_accumulated_offset;
    }
    __syncthreads();
}

PBRT_GPU void HLBVH::build_bottom_bvh(const BottomBVHArgs *bvh_args_array, uint array_length) {
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

        for (uint morton_idx = mid_idx; morton_idx < node.first_primitive_idx + node.num_primitives;
             morton_idx++) {
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

    build_nodes[args.build_node_idx].init_interior(split_dimension, left_child_idx, right_child_idx,
                                                   left_bounds + right_bounds);
}

PBRT_GPU bool HLBVH::fast_intersect(const Ray &ray, double t_max) const {
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

PBRT_GPU std::optional<ShapeIntersection> HLBVH::intersect(const Ray &ray, double t_max) const {
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

void HLBVH::build_bvh(std::vector<void *> &gpu_dynamic_pointers,
                      const std::vector<const Shape *> &gpu_primitives) {
    printf("\n");

    auto start_bvh = std::chrono::system_clock::now();

    uint num_total_primitives = gpu_primitives.size();

    printf("total primitives: %u\n", num_total_primitives);

    MortonPrimitive *gpu_morton_primitives;
    checkCudaErrors(cudaMallocManaged((void **)&gpu_morton_primitives,
                                      sizeof(MortonPrimitive) * num_total_primitives));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    const Shape **gpu_primitives_array;
    checkCudaErrors(
        cudaMallocManaged((void **)&gpu_primitives_array, sizeof(Shape *) * num_total_primitives));
    checkCudaErrors(cudaMemcpy(gpu_primitives_array, gpu_primitives.data(),
                               sizeof(Shape *) * num_total_primitives, cudaMemcpyHostToDevice));

    Treelet *sparse_treelets;
    checkCudaErrors(
        cudaMallocManaged((void **)&sparse_treelets, sizeof(Treelet) * MAX_TREELET_NUM));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    gpu_dynamic_pointers.push_back(gpu_morton_primitives);
    gpu_dynamic_pointers.push_back(gpu_primitives_array);

    this->init(gpu_primitives_array, gpu_morton_primitives);

    {
        uint threads = 512;
        uint blocks = divide_and_ceil(num_total_primitives, threads);
        hlbvh_init_morton_primitives<<<blocks, threads>>>(morton_primitives, primitives,
                                                          num_total_primitives);
    }
    {
        uint threads = 512;
        uint blocks = divide_and_ceil(MAX_TREELET_NUM, threads);
        hlbvh_init_treelets<<<blocks, threads>>>(sparse_treelets);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    Bounds3f bounds_of_primitives_centroids;
    for (int idx = 0; idx < num_total_primitives; idx++) {
        // TODO: make this one parallel (on CPU)?
        bounds_of_primitives_centroids += gpu_morton_primitives[idx].bounds.centroid();
    }

    {
        uint batch_size = 512;
        uint blocks = divide_and_ceil(num_total_primitives, batch_size);
        hlbvh_compute_morton_code<<<blocks, batch_size>>>(morton_primitives, num_total_primitives,
                                                          bounds_of_primitives_centroids);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    {
        struct {
            bool operator()(const MortonPrimitive &left, const MortonPrimitive &right) const {
                return (left.morton_code & TREELET_MASK) < (right.morton_code & TREELET_MASK);
            }
        } morton_comparator;

        std::sort(morton_primitives, morton_primitives + num_total_primitives, morton_comparator);
        // TODO: rewrite this sorting in GPU, check CUDA samples: radixSortThrust
        // https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/radixSortThrust
    }

    // init top treelets
    {
        uint threads = 512;
        uint blocks = divide_and_ceil(num_total_primitives, threads);

        hlbvh_collect_primitives_into_treelets<<<blocks, threads>>>(
            sparse_treelets, morton_primitives, primitives, num_total_primitives);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    std::vector<int> dense_treelet_indices;
    uint max_primitive_num_in_a_treelet = 0;

    {
        uint primitives_counter = 0;
        for (int idx = 0; idx < MAX_TREELET_NUM; idx++) {
            uint current_treelet_primitives_num = sparse_treelets[idx].n_primitives;
            if (current_treelet_primitives_num <= 0) {
                continue;
            }

            primitives_counter += current_treelet_primitives_num;

            max_primitive_num_in_a_treelet =
                std::max(max_primitive_num_in_a_treelet, current_treelet_primitives_num);
            dense_treelet_indices.push_back(idx);

            continue;
            printf("treelet[%u]: %u\n", dense_treelet_indices.size() - 1,
                   current_treelet_primitives_num);
        }
        printf("\n");

        assert(primitives_counter == num_total_primitives);
        printf("HLBVH: %zu/%d treelets filled (max primitives in a treelet: %d)\n",
               dense_treelet_indices.size(), MAX_TREELET_NUM, max_primitive_num_in_a_treelet);
    }

    Treelet *dense_treelets;
    checkCudaErrors(cudaMallocManaged((void **)&dense_treelets,
                                      sizeof(Treelet) * dense_treelet_indices.size()));

    for (int idx = 0; idx < dense_treelet_indices.size(); idx++) {
        int sparse_idx = dense_treelet_indices[idx];
        checkCudaErrors(cudaMemcpy(&dense_treelets[idx], &sparse_treelets[sparse_idx],
                                   sizeof(Treelet), cudaMemcpyDeviceToDevice));

        // printf("treelet[%d], num_primitives: %d\n", idx, dense_treelets[idx].n_primitives);
    }
    checkCudaErrors(cudaFree(sparse_treelets));

    uint max_build_node_length =
        (2 * dense_treelet_indices.size() + 1) + (2 * num_total_primitives + 1);
    checkCudaErrors(
        cudaMallocManaged((void **)&build_nodes, sizeof(BVHBuildNode) * max_build_node_length));
    gpu_dynamic_pointers.push_back(build_nodes);

    uint top_bvh_node_num =
        build_top_bvh_for_treelets(dense_treelet_indices.size(), dense_treelets);
    checkCudaErrors(cudaFree(dense_treelets));

    uint start = 0;
    uint end = top_bvh_node_num;

    uint *accumulated_offset;
    checkCudaErrors(cudaMallocManaged((void **)&accumulated_offset, sizeof(uint)));

    uint depth = 0;
    while (end > start) {
        const uint array_length = end - start;

        BottomBVHArgs *bvh_args_array;
        checkCudaErrors(
            cudaMallocManaged((void **)&bvh_args_array, sizeof(BottomBVHArgs) * array_length));

        *accumulated_offset = end;
        init_bvh_args<<<1, 1024>>>(bvh_args_array, accumulated_offset, build_nodes, start, end);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        printf("HLBVH: building bottom BVH: depth %u, leaves' number: %u\n", depth, array_length);

        depth += 1;
        start = end;
        end = *accumulated_offset;

        uint threads = 512;
        uint blocks = divide_and_ceil(array_length, threads);

        hlbvh_build_bottom_bvh<<<blocks, threads>>>(this, bvh_args_array, array_length);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(bvh_args_array));
    }
    checkCudaErrors(cudaFree(accumulated_offset));

    printf("HLBVH: bottom BVH max depth: %u (max primitives in a leaf: %u)\n", depth,
           MAX_PRIMITIVES_NUM_IN_LEAF);
    printf("HLBVH: total nodes: %u/%u\n", end, max_build_node_length);

    const std::chrono::duration<double> duration_bvh{std::chrono::system_clock::now() - start_bvh};
    std::cout << std::fixed << std::setprecision(2) << "BVH constructing took "
              << duration_bvh.count() << " seconds.\n"
              << std::flush;
}

uint HLBVH::build_top_bvh_for_treelets(uint num_dense_treelets, const Treelet *treelets) {
    std::vector<uint> treelet_indices;
    for (uint idx = 0; idx < num_dense_treelets; idx++) {
        treelet_indices.push_back(idx);
    }

    uint build_node_count = 0;
    uint max_depth = 0;
    recursive_build_top_bvh_for_treelets(treelet_indices, treelets, build_node_count, 0, max_depth);

    printf("HLBVH: top BVH nodes: %u, max depth: %u\n", build_node_count, max_depth);

    return build_node_count;
}

uint HLBVH::recursive_build_top_bvh_for_treelets(const std::vector<uint> &treelet_indices,
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
uint HLBVH::partition_morton_primitives(const uint start, const uint end,
                                        const uint8_t split_dimension, const double split_val) {
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
