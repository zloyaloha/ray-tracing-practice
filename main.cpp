#include <iostream>

#include <hittable_list.hpp>
#include <camera.hpp>
#include <sphere.hpp>

int main() {
    HittableList space;
    Camera cam(16.0 / 9.0, 400);
    space.push_back(std::make_shared<Sphere>(point3(0,0,-1), 0.6));
    // space.push_back(std::make_shared<Sphere>(point3(0,-100,-1), 99));

    cam.render(space);
}