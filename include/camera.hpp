#pragma once

#include <fstream>
#include <hittable_object.hpp>
#include <iostream>
#include <ray.hpp>

class ISaver {
protected:
    int sampelsPerPixel;
    int imageWidth;
    int imageHeight;

public:
    ISaver(int sampelsPerPixel);
    virtual ~ISaver() = default;
    virtual void writeColor(color pixel_color) = 0;
    double linearToGamma(const double &linear);
    virtual void setFormat(int imageWidth, int imageHeight) = 0;
};

class FileSaver : public ISaver {
public:
    FileSaver(int sampelsPerPixel, const std::string &filename);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;

private:
    std::ofstream fout;
};

class OutStreamSaver : public ISaver {
public:
    OutStreamSaver(int sampelsPerPixel);
    void writeColor(color pixel_color) override;
    void setFormat(int imageWidth, int imageHeight) override;
};

class Camera {
private:
    point3 coords;
    vec3 delta_pixel_LR, delta_pixel_UD;
    point3 zeroPixelLoc;
    int imageHeight;
    std::unique_ptr<ISaver> saver;

public:
    int imageWidth = 100;
    double aspectRatio = 1.0;
    int sampelsPerPixel = 30;
    int max_depth = 30;

    Camera(double aspect_ratio, int imageWidth, std::unique_ptr<ISaver> &saver);
    void render(const HittableObject &world);
    color rayColor(const Ray &ray, int depth,
                   const HittableObject &world) const;
    static double linearToGamma(const double &linear);

private:
    vec3 pixelSampleSquare() const;
    Ray getRay(int i, int j) const;
};