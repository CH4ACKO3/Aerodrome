#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "y_atmosphere.h"
#include "F16_low.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

PYBIND11_MODULE(F16_low, m)
{
    py::class_<F16_low, F16, std::shared_ptr<F16_low>>(m, "F16_low")
        .def(py::init<>())
        .def(py::init<py::dict>(), py::arg("input_dict") = py::dict())
        .def("step", &F16_low::step)
        .def("reset", &F16_low::reset);
}
