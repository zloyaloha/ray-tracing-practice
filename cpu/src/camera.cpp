#include <camera.hpp>
#include <iostream>

#include "different.hpp"
#include "vec3.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

ISaver::ISaver(int sampelsPerPixel) : sampelsPerPixel(sampelsPerPixel) {}

double ISaver::linearToGamma(const double &linear) { return std::sqrt(linear); }

void FileSaver::setFormat(int width, int height) {
    imageHeight = height;
    imageWidth = width;
    fout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
}

void OutStreamSaver::setFormat(int width, int height) {
    imageHeight = height;
    imageWidth = width;
    std::cout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
}

FileSaver::FileSaver(int sampelsPerPixel, const std::string &filesaver) : ISaver(sampelsPerPixel), fout(filesaver) {}

void FileSaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / sampelsPerPixel;

    double r = linearToGamma(pixel_color.x());
    double g = linearToGamma(pixel_color.y());
    double b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.000, 0.999);
    fout << static_cast<int>(256 * intensity.clamp(r)) << ' ' << static_cast<int>(256 * intensity.clamp(g)) << ' '
         << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

OutStreamSaver::OutStreamSaver(int sampelsPerPixel) : ISaver(sampelsPerPixel) {}

void OutStreamSaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / sampelsPerPixel;

    double r = linearToGamma(pixel_color.x());
    double g = linearToGamma(pixel_color.y());
    double b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.000, 0.999);
    std::cout << static_cast<int>(256 * intensity.clamp(r)) << ' ' << static_cast<int>(256 * intensity.clamp(g)) << ' '
              << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

PNGSaver::PNGSaver(int sampelsPerPixel, const std::string &filepath) : ISaver(sampelsPerPixel), filepath(filepath) {}

void PNGSaver::setFormat(int width, int height) {
    imageWidth = width;
    imageHeight = height;
    pixel_data.resize(width * height * 3, 0);
    pixel_count = 0;
}

void PNGSaver::writeColor(color pixel_color) {
    pixel_color = pixel_color / sampelsPerPixel;

    double r = linearToGamma(pixel_color.x());
    double g = linearToGamma(pixel_color.y());
    double b = linearToGamma(pixel_color.z());

    static const Interval intensity(0.0, 0.999);

    unsigned char r_byte = static_cast<unsigned char>(256 * intensity.clamp(r));
    unsigned char g_byte = static_cast<unsigned char>(256 * intensity.clamp(g));
    unsigned char b_byte = static_cast<unsigned char>(256 * intensity.clamp(b));

    int idx = pixel_count * 3;
    pixel_data[idx] = r_byte;
    pixel_data[idx + 1] = g_byte;
    pixel_data[idx + 2] = b_byte;

    pixel_count++;
}

PNGSaver::~PNGSaver() {
    if (!pixel_data.empty()) {
        int result = stbi_write_png(filepath.c_str(), imageWidth, imageHeight, 3, pixel_data.data(), imageWidth * 3);

        if (result) {
            std::cout << "✅ PNG saved: " << filepath << std::endl;
        } else {
            std::cerr << "❌ Failed to save PNG: " << filepath << std::endl;
        }
    }
}

Camera::Camera(double ratio, int width, std::unique_ptr<ISaver> &image_saver, const point3 &camera_pos,
               const point3 &look_at_point)
    : aspectRatio(ratio), imageWidth(width), saver(std::move(image_saver)), coords(camera_pos), look_at(look_at_point) {
    imageHeight = static_cast<int>(imageWidth / aspectRatio);
    imageHeight = (imageHeight < 1) ? 1 : imageHeight;
    saver->setFormat(imageWidth, imageHeight);

    vec3 world_up(0, 1, 0);

    forward = (look_at - coords).unit_vector();
    right = cross(forward, world_up).unit_vector();
    up = cross(right, forward).unit_vector();

    double vfov_rad = vfov * M_PI / 180.0;
    double focal_length = 1.0;
    double viewport_height = 2.0 * std::tan(vfov_rad / 2.0);
    double viewport_width = viewport_height * (static_cast<double>(imageWidth) / imageHeight);

    vec3 viewport_right = viewport_width * right;
    vec3 viewport_up = viewport_height * up;

    delta_pixel_LR = viewport_right / imageWidth;
    delta_pixel_UD = -viewport_up / imageHeight;

    vec3 viewport_upper_left = coords + focal_length * forward - viewport_right / 2.0 + viewport_up / 2.0;
    zeroPixelLoc = viewport_upper_left + (delta_pixel_LR + delta_pixel_UD) * 0.5;
}

void Camera::render(const HittableObject &world) {
    int counter = 0;
    for (size_t j = 0; j < imageHeight; ++j) {
        std::clog << "\rProgress bar: " << (imageHeight - j) << ' ' << std::flush;
        for (size_t i = 0; i < imageWidth; ++i) {
            color pixelColor(0, 0, 0);
            for (size_t sample = 0; sample < sampelsPerPixel; ++sample) {
                Ray r = getRay(i, j);
                pixelColor += rayColor(r, max_depth, world);
            }
            saver->writeColor(pixelColor);
        }
    }
    std::clog << "\rDone.                 \n";
}

color Camera::rayColor(const Ray &ray, int depth, const HittableObject &world) const {
    HitRecord rec;
    if (depth <= 0) {
        return color(0, 0, 0);
    }
    if (!world.hit(ray, Interval(1e-3, infinity), rec)) {
        return background_color;
    }

    Ray scattered;
    color attenuation;
    color color_from_emission = rec.material->emitted();

    if (!rec.material->scatter(ray, rec, attenuation, scattered)) {
        return color_from_emission;
    }
    return color_from_emission + attenuation * rayColor(scattered, depth - 1, world);
}

vec3 Camera::pixelSampleSquare() const {
    double dx = -0.5 + randomDouble(), dy = -0.5 + randomDouble();
    return (dx * delta_pixel_LR) + (dy * delta_pixel_UD);
}

Ray Camera::getRay(int i, int j) const {
    point3 pixel_loc = zeroPixelLoc + (i * delta_pixel_LR) + (j * delta_pixel_UD);
    point3 someRandomPixel = pixel_loc + pixelSampleSquare();

    return Ray(coords, someRandomPixel - coords);
}