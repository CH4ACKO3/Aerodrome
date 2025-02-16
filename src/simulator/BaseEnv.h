#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class BaseEnv
{
public:
    BaseEnv() = default;
    virtual ~BaseEnv() = default;

    virtual py::object reset() = 0;
    virtual py::object step(const py::object& action) = 0;
};
