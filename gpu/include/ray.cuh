#pragma once

#include "vec3.cuh"

class Ray {
public:
    inline __host__ __device__ Ray() : orig(), dir() {}
    inline __host__ __device__ Ray(const point3 &origin, const vec3 &direction) : orig(origin), dir(direction) {}

    inline __host__ __device__ point3 origin() const { return orig; }
    inline __host__ __device__ vec3 direction() const { return dir; }
    inline __host__ __device__ point3 at(float t) const { return orig + t * dir; }

private:
    point3 orig;
    vec3 dir;
};