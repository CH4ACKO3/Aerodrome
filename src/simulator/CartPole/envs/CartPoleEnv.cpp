#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "CartPoleEnv.h"
#include <string>
#include <cmath>
#include <random>

namespace py = pybind11;

PYBIND11_MODULE(CartPoleEnv, m)
{
    py::class_<CartPoleEnv, BaseEnv>(m, "CartPoleEnv")
        .def(py::init<>())
        .def("reset", &CartPoleEnv::reset)
        .def("step", &CartPoleEnv::step);
}
