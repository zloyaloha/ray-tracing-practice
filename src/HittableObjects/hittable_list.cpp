#include "hittable_list.hpp"

void HittableList::push_back(std::shared_ptr<HittableObject> obj) {
    _objs.push_back(obj);
}

void HittableList::clear() {
    _objs.clear();
}

bool HittableList::hit(const Ray& r,const double &ray_tmin, const double &ray_tmax, HitRecord& rec) const {
    HitRecord tmpRec;
    bool hitAny = false;
    auto closest_so_far = ray_tmax;
    for (const auto &obj: _objs) {
        if (obj->hit(r, ray_tmin, ray_tmax, tmpRec)) {
            hitAny = true;
            closest_so_far = tmpRec.param;
            rec = tmpRec;
        }
    }
    return hitAny;
}