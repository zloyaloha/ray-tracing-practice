#include <memory>
#include <plane.hpp>

Plane::Plane(const point3 &base, double radius, const vec3 &normal,
             std::shared_ptr<Material> mat)
    : _base(base),
      _normal(normal.unit_vector()),
      _material(mat),
      _radius(radius) {}

bool Plane::hit(const Ray &r, const Interval &ray_inter, HitRecord &rec) const {
  double denom = dot(_normal, r.direction().unit_vector());
  if (denom < 1e3) {
    return false;
  }
  auto root = dot(_base - r.origin(), _normal) / denom;
  if (!ray_inter.contains(root)) {
    return false;
  }
  rec.param = root;
  rec.point = r.at(root);
  vec3 vector = rec.point - _base;
  rec.set_orientation(r, _normal);
  rec.material = _material;
  return dot(vector, vector) < _radius * _radius;
}