#include <iostream>
#include <color.hpp>
#include <ray.hpp>
#include <sphere.hpp>
#include <hittable_list.hpp>
#include <constants.hpp>

color rayColor(const Ray& r, const HittableObject &world) {
    HitRecord rec;
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(0.3,0.5,0.3));
    }

    vec3 unit_dir = r.direction().unit_vector();
    auto a = 0.5*(unit_dir.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}


int main() {

    HittableList space;
    space.push_back(std::make_shared<Sphere>(point3(0,0.2,-1), 0.5));
    space.push_back(std::make_shared<Sphere>(point3(-3, -3, -5), 0.6));

    double aspect_ratio = 16.0 / 9.0;
    int imageWidth = 1920, imageHeight = static_cast<int>(imageWidth / aspect_ratio);
    imageHeight = (imageHeight < 1) ? 1 : imageHeight;
    
    double focal_lenght = 1.0;
    auto viewportHeight = 2.0, viewportWidth = viewportHeight * (static_cast<double> (imageWidth) / imageHeight);
    auto camera_center = point3(0,0,0);

    auto viewport_LR = vec3(viewportWidth, 0, 0), viewport_UD = vec3(0, -viewportHeight, 0);

    auto delta_pixel_LR = viewport_LR / imageWidth, delta_pixel_UD = viewport_UD / imageHeight;
    auto viewport_upper_left = camera_center - vec3(0,0,focal_lenght) - viewport_LR / 2 - viewport_UD / 2;
    auto pixel00_loc = viewport_upper_left + (delta_pixel_LR + delta_pixel_UD) / 2;

    std::cout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
    int counter = 0;
    for (size_t j = 0; j < imageHeight; ++j) {
        std::clog << "\rProgress bar: " << (imageHeight - j) << ' ' << std::flush;
        for (size_t i = 0; i < imageWidth; ++i) {
            auto pixel_loc = pixel00_loc + (i * delta_pixel_LR) + (j * delta_pixel_UD);
            auto ray_dir = pixel_loc - camera_center;

            Ray r(camera_center, ray_dir);
            color pixel_color = rayColor(r, space);
            write_color(std::cout, pixel_color);
        }
    }
    std::clog << "\rDone.                 \n";
}