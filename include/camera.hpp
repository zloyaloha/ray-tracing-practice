#pragma once

#include <iostream>
#include <hittable_object.hpp>

using color = vec3;

class Camera {
    private:
        point3 coords;
        vec3 delta_pixel_LR, delta_pixel_UD;
        point3 zeroPixelLoc;
        int imageHeight;
    public:
        int imageWidth = 1920;
        double aspectRatio = 16.0 / 9.0;
        int sampelsPerPixel = 15;
        int max_depth = 15;
        Camera() = default;
        Camera(double aspect_ratio, int imageWidth);
        void render(const HittableObject &world);
        color rayColor(const Ray &ray, int depth, const HittableObject &world) const;
        void writeColor(std::ostream &out, color pixel_color);
    private:
        vec3 pixelSampleSquare() const;
        Ray getRay(int i, int j) const;
};