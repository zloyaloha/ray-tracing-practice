#pragma once

#include <iostream>
#include <hittable_object.hpp>
#include <ray.hpp>

class Camera {
    private:
        point3 coords;
        vec3 delta_pixel_LR, delta_pixel_UD;
        point3 zeroPixelLoc;
        int imageHeight;
    public:
        int imageWidth = 100;
        double aspectRatio = 1.0;
        int sampelsPerPixel = 30;
        int max_depth = 30;

        Camera(double aspect_ratio, int imageWidth);
        void render(const HittableObject &world);
        color rayColor(const Ray &ray, int depth, const HittableObject &world) const;
        void writeColor(std::ostream &out, color pixel_color);
        static double linearToGamma(const double &linear);
    private:
        vec3 pixelSampleSquare() const;
        Ray getRay(int i, int j) const;
};