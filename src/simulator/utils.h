#ifndef UTILS_H_
#define UTILS_H_

#include <cmath>

namespace utils {
    double norm(const std::vector<double>& vec) {
        return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    }

    double norm(const std::array<double, 3>& vec) {
        return std::sqrt(std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0));
    }
}

#endif

