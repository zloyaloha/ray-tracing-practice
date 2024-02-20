#include <iostream>

#include <hittable_list.hpp>
#include <camera.hpp>
#include <sphere.hpp>
#include <lambertian.hpp>

int main() {
    HittableList space;
    Camera cam(16.0 / 9.0, 400);
    cam.sampelsPerPixel = 30;
    cam.max_depth = 30;

    auto material_ground = std::make_shared<Lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<Lambertian>(color(0.7, 0.3, 0.3));

    space.push_back(std::make_shared<Sphere>(point3(0,0,-1), 0.5, material_center));
    space.push_back(std::make_shared<Sphere>(point3(0,-100.5,-1), 100.4, material_ground));

    cam.render(space);
}