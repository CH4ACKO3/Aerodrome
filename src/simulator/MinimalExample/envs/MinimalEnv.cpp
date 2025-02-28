#include "MinimalEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(MinimalEnv, m) {
    py::class_<MinimalEnv, BaseEnv>(m, "MinimalEnv")
        .def(py::init<>())
        .def("reset", &MinimalEnv::reset)
        .def("step", &MinimalEnv::step);
}
