#pragma once

#include <curand_kernel.h>

#include "vec3.cuh"

__device__ inline unsigned int wang_hash(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

__device__ inline float random_float(unsigned int &seed) {
    seed = wang_hash(seed);
    return static_cast<float>(seed) / 4294967296.0f;
}

__device__ inline float random_float(unsigned int &seed, float min, float max) { return min + (max - min) * random_float(seed); }

__device__ inline vec3 random_in_unit_sphere(unsigned int &seed) {
    while (true) {
        vec3 candidate(random_float(seed, -1.0, 1.0), random_float(seed, -1.0, 1.0), random_float(seed, -1.0, 1.0));
        if (candidate.len_squared() < 1.0) {
            return candidate;
        }
    }
}

__device__ inline vec3 random_unit_vector(unsigned int &seed) { return unit_vector(random_in_unit_sphere(seed)); }

__device__ inline vec3 random_in_hemisphere(const vec3 &normal, unsigned int &seed) {
    vec3 in_sphere = random_unit_vector(seed);
    if (dot(in_sphere, normal) > 0.0) {
        return in_sphere;
    }
    return -in_sphere;
}
