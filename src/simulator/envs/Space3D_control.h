#pragma once

#include "Space3D.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

class Space3D_control : public Space3D
{
public:
    Space3D_control(double tau, double eps, std::string integrator, int integrate_steps): Space3D(tau, eps, integrator, integrate_steps) {};

    virtual py::object reset() override
    {   
        return py::none();
    }

    virtual py::object step(const py::object& action) override
    {
        return py::none();
    }
};
