#pragma once
#include "constants.hpp"

class Interval {
    private:
        double min, max;
    public:
        Interval() : min(-infinity), max(+infinity) {}
        Interval(const double &min_, const double &max_) : min(min_), max(max_) {}

        bool contains(const double &x) const{
            return min <= x && x <= max;
        }

        bool surrounds(const double &x) const {
            return min < x && x < max;
        }

        double clamp(double x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }

        double getRight() const { return max; }
        double getLeft() const { return min; }

};