#pragma once
#include "different.hpp"

class Interval {
    private:
        double min, max;
    public:
        Interval();
        Interval(const double &min_, const double &max_);

        bool contains(const double &x) const;
        bool surrounds(const double &x) const;
        double clamp(double x) const;

        double getRight() const;
        double getLeft() const;
};