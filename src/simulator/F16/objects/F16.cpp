#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "y_atmosphere.h"
#include "F16.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

PYBIND11_MODULE(F16, m)
{
    py::class_<F16, Object3D, std::shared_ptr<F16>>(m, "F16")
        .def(py::init<>())
        .def(py::init<py::dict>(), py::arg("input_dict") = py::dict())
        .def("step", &F16::step)
        .def("reset", &F16::reset);
}
