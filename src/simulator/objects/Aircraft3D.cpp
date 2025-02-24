#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Aircraft3D.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

PYBIND11_MODULE(Aircraft3D, m)
{
    py::class_<Aircraft3D, Object3D, std::shared_ptr<Aircraft3D>>(m, "Aircraft3D")
        .def(py::init<py::dict>(), py::arg("input_dict") = py::dict())
        .def(py::init<py::dict>())
        .def("to_dict", &Aircraft3D::to_dict)
        .def("d", &Aircraft3D::d)
        .def("update", &Aircraft3D::update);
}