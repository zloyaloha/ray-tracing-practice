#include <iostream>

#include <hittable_list.hpp>
#include <camera.hpp>
#include <sphere.hpp>

int main() {
    HittableList space;
    Camera cam(16.0 / 9.0, 400);
    cam.sampelsPerPixel = 30;
    cam.max_depth = 30;
    space.push_back(std::make_shared<Sphere>(point3(0,0,-1), 0.5));
    space.push_back(std::make_shared<Sphere>(point3(0,-100.5,-1), 100.4));

    cam.render(space);
}