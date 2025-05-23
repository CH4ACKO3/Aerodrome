#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "y_atmosphere.h"
#include "WingedCone2D.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

PYBIND11_MODULE(WingedCone2D, m)
{
    py::class_<WingedCone2D, Object3D, std::shared_ptr<WingedCone2D>>(m, "WingedCone2D")
        .def(py::init<>())
        .def(py::init<py::dict>(), py::arg("input_dict") = py::dict())
        .def("step", &WingedCone2D::step)
        .def("reset", &WingedCone2D::reset);
}

