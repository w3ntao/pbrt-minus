#include "pbrt/util/hash_map.h"

// taken from
// https://lemire.me/blog/2018/08/15/fast-strongly-universal-64-bit-hashing-everywhere/
PBRT_CPU_GPU
uint64_t hash(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53L;
    k ^= k >> 33;

    return k;
}

HashMap *HashMap::create(uint capacity, std::vector<void *> &gpu_dynamic_pointers) {
    HashMap *hash_map;
    CHECK_CUDA_ERROR(cudaMallocManaged(&hash_map, sizeof(HashMap)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&(hash_map->items), sizeof(KeyValue) * capacity));

    gpu_dynamic_pointers.push_back(hash_map);
    gpu_dynamic_pointers.push_back(hash_map->items);

    hash_map->size = 0;
    hash_map->capacity = capacity;

    for (uint idx = 0; idx < capacity; ++idx) {
        hash_map->items[idx].key = EMPTY_KEY;
    }

    return hash_map;
}

PBRT_CPU_GPU
uint64_t HashMap::lookup(uint64_t key) const {
    uint64_t slot = hash(key) % capacity;

    while (true) {
        if (items[slot].key == key) {
            return items[slot].value;
        }

        slot = (slot + 1) % capacity;
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
void HashMap::insert(uint64_t key, uint64_t value) {
    if (size >= capacity) {
        REPORT_FATAL_ERROR();
    }

    size += 1;

    uint64_t slot = hash(key) % capacity;

    while (true) {
        if (items[slot].key == key) {
            printf("%s(): %lu key was already taken\n", __func__, key);
            REPORT_FATAL_ERROR();
        }

        if (items[slot].key == EMPTY_KEY) {
            items[slot].key = key;
            items[slot].value = value;
            return;
        }

        slot = (slot + 1) % capacity;
    }
}
