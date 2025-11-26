#include "plain.cuh"

struct __align__(16) OctaederData {
    PlainData *planes;
    __host__ OctaederData::OctaederData(const point3 &base, const vec3 &u, const vec3 &v, int material_index, PlaneType type);
};
