#include <camera.hpp>
#include <hittable_list.hpp>
#include <lambertian.hpp>
#include <memory>
#include <sphere.hpp>

#include "bvh.hpp"
#include "dielectric.hpp"
#include "diffuse_light.hpp"
#include "metal.hpp"
#include "plane.hpp"
#include "vec3.hpp"

std::shared_ptr<HittableList> box_tetrahedron(const point3& center, double scale, std::shared_ptr<Material> mat) {
    auto sides = std::make_shared<HittableList>();

    point3 p0 = center + vec3(1, 1, 1) * scale;
    point3 p1 = center + vec3(1, -1, -1) * scale;
    point3 p2 = center + vec3(-1, 1, -1) * scale;
    point3 p3 = center + vec3(-1, -1, 1) * scale;

    sides->push_back(std::make_shared<Triangle>(p0, p1 - p0, p2 - p0, mat));
    sides->push_back(std::make_shared<Triangle>(p0, p2 - p0, p3 - p0, mat));
    sides->push_back(std::make_shared<Triangle>(p0, p3 - p0, p1 - p0, mat));
    sides->push_back(std::make_shared<Triangle>(p1, p3 - p1, p2 - p1, mat));

    return sides;
}

int main() {
    const int num_frames = 1;
    const double fps = 1.0;

    for (int frame = 0; frame < num_frames; ++frame) {
        HittableList space;

        auto glass = std::make_shared<Dielectric>(1.5);
        auto lambertian3 = std::make_shared<Lambertian>(color(0.1, 1, 0.1));
        auto lambertian2 = std::make_shared<Lambertian>(color(0.1, 0.1, 0.1));
        auto lambertian1 = std::make_shared<Lambertian>(color(0.7, 0.3, 0.3));
        auto metal = std::make_shared<Metal>(color(0.3, 0.3, 1), 0.4);
        auto light_mat = std::make_shared<DiffuseLight>(color(3.0, 3.0, 3.0));

        double r = 15.0;  // Радиус
        double d = r * 2.0;
        space.push_back(box_tetrahedron(point3(0, 1, 3), 0.5, lambertian3));
        space.push_back(std::make_shared<Ellipse>(point3(-r, 0, -r), vec3(d, 0, 0), vec3(0, 0, d), lambertian2));
        space.push_back(std::make_shared<Quad>(point3(-1, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), metal));
        space.push_back(std::make_shared<Sphere>(point3(1, 1, 1), 0.5, glass));
        space.push_back(std::make_shared<Sphere>(point3(1, 1, 2), 0.5, metal));
        space.push_back(std::make_shared<Sphere>(point3(0, 1, 1), 0.5, lambertian1));
        space.push_back(std::make_shared<Sphere>(point3(10, 10, 10), 5.0, light_mat));
        BVHNode bvh(space);

        double t = frame / static_cast<double>(num_frames);
        double distance = -5.0 + 5.0 * t;
        point3 camera_pos(3, 1, 3);
        std::unique_ptr<ISaver> saver = std::make_unique<PNGSaver>(30, "../images/render" + std::to_string(t) + ".png");
        Camera cam(16.0 / 9.0, 480, saver, camera_pos);
        cam.sampelsPerPixel = 10;
        cam.max_depth = 1000;

        cam.render(bvh);
    }
}