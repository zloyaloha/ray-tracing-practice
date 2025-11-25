#pragma once

#include <cmath>

constexpr float kInfinity = 1e32;

class Interval {
public:
    float min;
    float max;

    __host__ __device__ Interval() : min(+kInfinity), max(-kInfinity) {}
    __host__ __device__ Interval(float min_value, float max_value) : min(min_value), max(max_value) {}

    __host__ __device__ bool contains(float value) const { return min <= value && value <= max; }
    __host__ __device__ bool surrounds(float value) const { return min < value && value < max; }
    __host__ __device__ float clamp(float value) const {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    __host__ __device__ float length() const { return max - min; }
};

static const Interval interval_empty(+kInfinity, -kInfinity);
static const Interval interval_universe(-kInfinity, +kInfinity);