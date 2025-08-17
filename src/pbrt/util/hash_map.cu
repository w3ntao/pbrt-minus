#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/util/hash_map.h>

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

HashMap::HashMap(const int _capacity, GPUMemoryAllocator &allocator) : capacity(_capacity) {
    items = allocator.allocate<KeyValue>(capacity);
    for (int idx = 0; idx < capacity; ++idx) {
        items[idx].key = EMPTY_KEY;
    }
}

PBRT_CPU_GPU
uint64_t HashMap::lookup(const uint64_t key) const {
    uint64_t slot = hash(key) % capacity;

    for (int loop = 0; loop < capacity; ++loop) {
        if (items[slot].key == key) {
            return items[slot].value;
        }

        slot = (slot + 1) % capacity;
    }

    REPORT_FATAL_ERROR();
    return SIZE_MAX;
}

PBRT_CPU_GPU
void HashMap::insert(const uint64_t key, const uint64_t value) {
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
