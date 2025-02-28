#pragma once

#include "BaseEnv.h"
#include "Object3D.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <iostream>

namespace py = pybind11;

class Space3D : public BaseEnv
{
private:
    double tau;
    double dt;
    double eps;
    int integrate_steps;

public:
    std::vector<std::shared_ptr<Object3D>> objects;

    Space3D(double tau, double eps, int integrate_steps)
        : tau(tau), eps(eps), integrate_steps(integrate_steps)
    {
        dt = tau / integrate_steps;
    }

    void add_object(std::shared_ptr<Object3D> object)
    {
        objects.push_back(object);
    }

    bool check_consistency()
    {
        return true;
    }

    py::dict to_dict()
    {
        py::dict output_dict;
        for (auto& object : objects)
        {
            output_dict[py::str(object->name)] = object->to_dict();
        }
        return output_dict;
    }

    py::dict get_d()
    {
        py::dict output_dict;
        for (auto& object : objects)
        {
            output_dict[py::str(object->name)] = (object->d()).to_dict();
        }
        return output_dict;
    }

    py::object step(const py::object& actions)
    {
        py::dict result, info;

        for (int i = 0; i < integrate_steps; ++i)
        {
            for (auto& object : objects)
            {
                actions[py::str(object->name)]["dt"] = dt;   
                result[py::str(object->name)] = object->step(actions[py::str(object->name)]);
            }
        }

        return result;
    }
};