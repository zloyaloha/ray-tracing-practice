#pragma once

#include "hittable_object.hpp"
#include <vector>
#include <memory>

class HittableList : public HittableObject {
    private:
        std::vector<std::shared_ptr<HittableObject>> _objs;
    public:
        HittableList() = default;
        
        void push_back(std::shared_ptr<HittableObject> obj);
        void clear();

        bool hit(const Ray& r,const double &ray_tmin, const double &ray_tmax, HitRecord& rec) const override;
};