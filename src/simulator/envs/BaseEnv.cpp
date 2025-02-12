#include "BaseEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(BaseEnv, m) {
    py::class_<BaseEnv>(m, "BaseEnv")
        .def("reset", &BaseEnv::reset)
        .def("step", &BaseEnv::step);
}
