#include "Object3D.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

PYBIND11_MODULE(Object3D, m)
{
    py::class_<Object3D, std::shared_ptr<Object3D>>(m, "Object3D")
        .def(py::init<py::dict>(), py::arg("input_dict") = py::dict())
        .def(py::init<py::dict>())
        .def("to_dict", &Object3D::to_dict)
        .def("__add__", &Object3D::operator+)
        .def("__sub__", &Object3D::operator-)
        .def("__mul__", &Object3D::operator*)
        .def("__truediv__", &Object3D::operator/);
}
