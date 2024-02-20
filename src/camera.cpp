#include <camera.hpp>
#include <interval.hpp>

Camera::Camera(double ratio, int width) : aspectRatio(ratio), imageWidth(width) {
    imageHeight = static_cast<int>(imageWidth / aspectRatio);
    imageHeight = (imageHeight < 1) ? 1 : imageHeight;
    
    double focal_lenght = 1.0;
    auto viewportHeight = 2.0, viewportWidth = viewportHeight * (static_cast<double> (imageWidth) / imageHeight);
    coords = point3(0,0,0);

    vec3 viewport_LR = vec3(viewportWidth, 0, 0), viewport_UD = vec3(0, -viewportHeight, 0);
    delta_pixel_LR = viewport_LR / imageWidth, delta_pixel_UD = viewport_UD / imageHeight;
    vec3 viewport_upper_left = coords - vec3(0, 0, focal_lenght) - viewport_LR / 2 - viewport_UD / 2;
    zeroPixelLoc = viewport_upper_left + (delta_pixel_LR + delta_pixel_UD) / 2;
}

void Camera::writeColor(std::ostream &out, color pixel_color) {

    pixel_color = pixel_color / sampelsPerPixel;

    double r = linearToGamma(pixel_color.x());
    double g = linearToGamma(pixel_color.y());
    double b = linearToGamma(pixel_color.z());
    
    static const Interval intensity(0.000, 0.999);
    out << static_cast<int>(256 * intensity.clamp(r)) << ' '
        << static_cast<int>(256 * intensity.clamp(g)) << ' '
        << static_cast<int>(256 * intensity.clamp(b)) << '\n';
}

void Camera::render(const HittableObject &world) {
    std::cout << "P3\n" << imageWidth << ' ' << imageHeight << "\n255\n";
    int counter = 0;
    for (size_t j = 0; j < imageHeight; ++j) {
        std::clog << "\rProgress bar: " << (imageHeight - j) << ' ' << std::flush;
        for (size_t i = 0; i < imageWidth; ++i) {
            color pixelColor(0,0,0);
            for (size_t sample = 0; sample < sampelsPerPixel; ++sample) {
                Ray r = getRay(i, j);
                pixelColor += rayColor(r, max_depth, world);
            }
            writeColor(std::cout, pixelColor);
        }
    }
    std::clog << "\rDone.                 \n";
}

color Camera::rayColor(const Ray &ray, int depth, const HittableObject &world) const {
    HitRecord rec;
    if (depth <= 0) {
        return color(0,0,0);
    }
    if (world.hit(ray, Interval(0.000001, infinity), rec)) {
        vec3 direction = rec.normal + vec3::randomUnitVectorInSphere();
        color tmp = 0.5 * rayColor(Ray(rec.point, direction), depth - 1, world);
        return tmp;
    }
    vec3 unit_dir = ray.direction().unit_vector();
    auto a = 0.2*(unit_dir.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
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

double Camera::linearToGamma(const double &linear) {
    return std::sqrt(linear);
}