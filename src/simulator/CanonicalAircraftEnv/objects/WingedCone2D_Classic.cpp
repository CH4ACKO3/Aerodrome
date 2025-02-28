#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "WingedCone2D_Classic.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

PYBIND11_MODULE(WingedCone2D_Classic, m)
{
    py::class_<WingedCone2D_Classic, WingedCone2D, std::shared_ptr<WingedCone2D_Classic>>(m, "WingedCone2D_Classic")
        .def(py::init<>())
        .def(py::init<py::dict>(), py::arg("input_dict") = py::dict())
        .def("step", &WingedCone2D_Classic::step);
}
