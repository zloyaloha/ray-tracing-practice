#include "interval.hpp"

Interval::Interval() : min(-infinity), max(+infinity) {}

Interval::Interval(const double &min_, const double &max_) : min(min_), max(max_) {}

bool Interval::contains(const double &x) const{
    return min <= x && x <= max;
}

bool Interval::surrounds(const double &x) const {
    return min < x && x < max;
}

double Interval::clamp(double x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

double Interval::getRight() const { 
    return max; 
}

double Interval::getLeft() const { 
    return min; 
}