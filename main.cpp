#include <iostream>

#include <hittable_list.hpp>
#include <camera.hpp>
#include <sphere.hpp>
#include <lambertian.hpp>
#include "metal.hpp"
#include "plane.hpp"
#include "dielectric.hpp"

int main() {
    HittableList space;
    Camera cam(16.0 / 9.0, 1080);
    cam.sampelsPerPixel = 50;
    cam.max_depth = 50;

    auto glass = std::make_shared<Dielectric>(1.5);
    auto lambertian2 = std::make_shared<Lambertian>(color(0.3, 0.4, 0.0));
    auto lambertian1 = std::make_shared<Lambertian>(color(0.7, 0.3, 0.3));
    auto metal = std::make_shared<Metal>(color(0.3, 0.3, 0.3), 0.4);

    space.push_back(make_shared<Sphere>(point3( 0.0, -100.5, -1.0), 100.0, lambertian2));
    space.push_back(std::make_shared<Sphere>(point3(0,0,-1), -0.5, glass));
    space.push_back(std::make_shared<Sphere>(point3(1,0,-1), 0.5, metal));
    space.push_back(std::make_shared<Sphere>(point3(0,0,-10), 0.5, lambertian1));
    cam.render(space);
}