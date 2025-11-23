#include "hittable_list.hpp"

#include "aabb.hpp"
#include "hittable_object.hpp"

void HittableList::push_back(std::shared_ptr<HittableObject> obj) {
    _objs.push_back(obj);
    bbox = AABB(bbox, obj->bounding_box());
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

std::vector<std::shared_ptr<HittableObject>> HittableList::getObjects() const {
    return _objs;
}

std::vector<std::shared_ptr<HittableObject>>& HittableList::getObjects() {
    return _objs;
}