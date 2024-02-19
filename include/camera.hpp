#pragma once

#include <iostream>
#include <hittable_object.hpp>

using color = vec3;

class Camera {
    private:
        point3 coords;
        vec3 delta_pixel_LR, delta_pixel_UD;
        point3 zeroPixelLoc;
        int imageHeight, imageWidth;
        double aspectRatio;
        int sampelsPerPixel = 10;
    public:
        Camera(double aspect_ratio, int imageWidth);
        void render(const HittableObject &world);
        color rayColor(const Ray &ray, const HittableObject &world) const;
        void writeColor(std::ostream &out, color pixel_color);
    private:
        vec3 pixelSampleSquare() const;
        Ray getRay(int i, int j) const;
};