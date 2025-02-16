#include "Space3D.h"
#include "Object3D.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;



PYBIND11_MODULE(Space3D, m)
{
    py::class_<Space3D, BaseEnv>(m, "Space3D")
        .def(py::init<double, double, std::string, int>(), pybind11::arg("tau") = 0.01, pybind11::arg("eps") = 0.000001, pybind11::arg("integrator") = "euler", pybind11::arg("integrate_steps") = 1)
        .def("add_object", &Space3D::add_object)
        .def("check_consistency", &Space3D::check_consistency)
        .def("to_dict", &Space3D::to_dict)
        .def("kinematics_step", &Space3D::kinematics_step);
}
