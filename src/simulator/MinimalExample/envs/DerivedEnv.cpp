#include "DerivedEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(DerivedEnv, m) {
    py::class_<DerivedEnv, BaseEnv>(m, "DerivedEnv")
        .def(py::init<>())
        .def("reset", &DerivedEnv::reset)
        .def("step", &DerivedEnv::step);
}
