#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "BaseEnv.h"
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(BaseEnv, m) {
    py::class_<BaseEnv>(m, "BaseEnv")
        .def("reset", &BaseEnv::reset)
        .def("step", &BaseEnv::step);
}
