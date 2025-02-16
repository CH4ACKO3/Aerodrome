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
    Space3D_control(double tau = 0.01, double eps = 0.000001, std::string integrator = "euler", int integrate_steps = 1)
        : Space3D(tau, eps, integrator, integrate_steps)
    {
    }

    virtual py::object reset() override
    {   
        return py::none();
    }

    py::object step(const py::object& action) override
    {
        for (auto& object : objects)
        {
            action[py::str(object->name)]["dt"] = dt;
            object->step(action[py::str(object->name)]);
        }

        kinematics_step();

        py::dict obs = to_dict();
        
        return obs;
    }
};

PYBIND11_MODULE(Space3D_control, m)
{
    py::class_<Space3D_control, Space3D>(m, "Space3D_control")
        .def(py::init<double, double, std::string, int>(), 
            pybind11::arg("tau") = 0.01, 
            pybind11::arg("eps") = 0.000001,
            pybind11::arg("integrator") = "euler",
            pybind11::arg("integrate_steps") = 1)
        .def("add_object", &Space3D::add_object)
        .def("step", &Space3D_control::step);
}
