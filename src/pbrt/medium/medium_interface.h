#pragma once

struct Medium;

struct MediumInterface {
    const Medium *interior = nullptr;
    const Medium *exterior = nullptr;

    MediumInterface(const Medium *_interior, const Medium *_exterior)
        : interior(_interior), exterior(_exterior) {}

    PBRT_CPU_GPU
    bool is_medium_transition() const {
        return interior != exterior;
    }
};
