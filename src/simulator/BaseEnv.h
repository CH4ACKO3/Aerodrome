#ifndef BASE_ENV_H
#define BASE_ENV_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class BaseEnv {
public:
    BaseEnv() = default;
    virtual ~BaseEnv() = default;

    virtual py::object reset() = 0;
    virtual py::object step(const py::object& action) = 0;
};

#endif // BASE_ENV_H
