#include <memory>
#include <plane.cuh>

#include "hittable_object.cuh"
#include "interval.cuh"
#include "vec3.cuh"

__host__ __device__ Quad::Quad(const point3 &base, const vec3 &u, const vec3 &v, std::shared_ptr<Material> mat)
    : _material(mat), _base(base), _u(u), _v(v) {
    auto n = cross(u, v);

    _normal = n.unit_vector();

    D = dot(_normal, base);

    _w = n / dot(n, n);

    auto bbox_diagonal1 = AABB(base, base + u + v);
    auto bbox_diagonal2 = AABB(base + u, base + v);
    bbox = AABB(bbox_diagonal1, bbox_diagonal2);
}

__device__ bool Quad::hit(const Ray &r, const Interval &ray_inter, HitRecord &rec) const {
    float denom = dot(_normal, r.direction());
    if (std::abs(denom) < 1e-8) {
        return false;
    }
    // auto root = dot(_base - r.origin(), _normal) / denom;
    auto root = (D - dot(_normal, r.origin())) / denom;
    if (!ray_inter.contains(root)) {
        return false;
    }

    auto intersection = r.at(root);
    vec3 planar_hitpt_vector = intersection - _base;
    auto alpha = dot(_w, cross(planar_hitpt_vector, _v));
    auto beta = dot(_w, cross(_u, planar_hitpt_vector));

    if (!is_interior(alpha, beta, rec)) return false;

    rec.param = root;
    rec.point = r.at(root);
    vec3 vector = rec.point - _base;
    rec.set_orientation(r, _normal);
    rec.material = _material;

    return true;
}

__device__ bool Quad::is_interior(float a, float b, HitRecord &rec) const {
    Interval unit_interval = Interval(0, 1);

    if (!unit_interval.contains(a) || !unit_interval.contains(b)) return false;

    rec.u = a;
    rec.v = b;
    return true;
}

__host__ __device__ AABB Quad::bounding_box() const { return bbox; }

__host__ __device__ Ellipse::Ellipse(const point3 &base, const vec3 &u, const vec3 &v, std::shared_ptr<Material> mat)
    : Quad(base, u, v, mat) {}

__device__ bool Ellipse::is_interior(float a, float b, HitRecord &rec) const {
    if (std::pow(a - 0.5, 2) + std::pow(b - 0.5, 2) > 0.25) return false;

    rec.u = a;
    rec.v = b;
    return true;
}

__host__ __device__ Triangle::Triangle(const point3 &base, const vec3 &u, const vec3 &v, std::shared_ptr<Material> mat)
    : Quad(base, u, v, mat) {}

__device__ bool Triangle::is_interior(float a, float b, HitRecord &rec) const {
    if (a < 0 || b < 0 || (a + b) > 1) {
        return false;
    }
    rec.u = a;
    rec.v = b;
    return true;
}