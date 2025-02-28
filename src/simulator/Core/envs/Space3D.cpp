#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Space3D.h"
#include <vector>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(Space3D, m)
{
    py::class_<Space3D, BaseEnv>(m, "Space3D")
        .def(py::init<double, double, int>(), pybind11::arg("tau") = 0.01, pybind11::arg("eps") = 0.01,  pybind11::arg("integrate_steps") = 1)
        .def("add_object", &Space3D::add_object)
        .def("to_dict", &Space3D::to_dict)
        .def("get_d", &Space3D::get_d)
        .def("step", &Space3D::step);
}