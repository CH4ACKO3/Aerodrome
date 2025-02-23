#include "Space3D_control.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

PYBIND11_MODULE(Space3D_control, m)
{
    py::class_<Space3D_control, Space3D>(m, "Space3D_control")
        .def(py::init<double, double, std::string, int>(), pybind11::arg("tau") = 0.01, pybind11::arg("eps") = 0.000001, pybind11::arg("integrator") = "euler", pybind11::arg("integrate_steps") = 1)
        .def("reset", &Space3D_control::reset)
        .def("step", &Space3D_control::step);
}