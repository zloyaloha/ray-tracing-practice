#include "hittable_list.hpp"

void HittableList::push_back(std::shared_ptr<HittableObject> obj) {
    _objs.push_back(obj);
}

void HittableList::clear() { _objs.clear(); }

bool HittableList::hit(const Ray& r, const Interval& ray_inter,
                       HitRecord& rec) const {
    HitRecord tmpRec;
    bool hitAny = false;
    auto closest_so_far = ray_inter.getRight();
    for (const auto& obj : _objs) {
        if (obj->hit(r, Interval(ray_inter.getLeft(), closest_so_far),
                     tmpRec)) {
            hitAny = true;
            closest_so_far = tmpRec.param;
            rec = tmpRec;
        }
    }
    return hitAny;
}