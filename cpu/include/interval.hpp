#pragma once
#include "different.hpp"

class Interval {
public:
    double min, max;

public:
    Interval();
    Interval(const double &min_, const double &max_);
    Interval(const Interval &a, const Interval &b);

    bool contains(const double &x) const;
    bool surrounds(const double &x) const;
    double clamp(double x) const;
    double len() const;

    double getRight() const;
    double getLeft() const;
    Interval expand(double delta) const;
};

inline Interval::Interval() : min(-infinity), max(+infinity) {}

inline Interval::Interval(const Interval &a, const Interval &b)
    : min(std::min(a.min, b.min)), max(std::max(a.max, b.max)) {}

inline Interval::Interval(const double &min_, const double &max_)
    : min(min_), max(max_) {}

inline bool Interval::contains(const double &x) const {
    return min <= x && x <= max;
}

inline bool Interval::surrounds(const double &x) const {
    return min < x && x < max;
}

inline double Interval::len() const { return std::fabs(max - min); }

inline double Interval::clamp(double x) const {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline double Interval::getRight() const { return max; }

inline double Interval::getLeft() const { return min; }

inline Interval Interval::expand(double delta) const {
    return Interval(min - delta, max + delta);
}