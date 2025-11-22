#include <camera.hpp>
#include <hittable_list.hpp>
#include <lambertian.hpp>
#include <memory>
#include <sphere.hpp>

#include "dielectric.hpp"
#include "metal.hpp"

int main() {
    HittableList space;
    std::unique_ptr<ISaver> saver =
        std::make_unique<FileSaver>(30, "../render.ppm");
    Camera cam(16.0 / 9.0, 1080, saver);
    cam.sampelsPerPixel = 10;
    cam.max_depth = 10;

    auto glass = std::make_shared<Dielectric>(1.5);
    auto lambertian2 = std::make_shared<Lambertian>(color(0.3, 0.4, 0.0));
    auto lambertian1 = std::make_shared<Lambertian>(color(0.7, 0.3, 0.3));
    auto metal = std::make_shared<Metal>(color(0.3, 0.3, 0.3), 0.4);

    space.push_back(
        make_shared<Sphere>(point3(0.0, -100.5, -1.0), 100.0, lambertian2));
    space.push_back(std::make_shared<Sphere>(point3(0, 0, -1), -0.5, glass));
    space.push_back(std::make_shared<Sphere>(point3(1, 0, -1), 0.5, metal));
    space.push_back(
        std::make_shared<Sphere>(point3(0, 0, -10), 0.5, lambertian1));
    cam.render(space);
}