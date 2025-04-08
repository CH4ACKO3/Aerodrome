#pragma once

#include "BaseEnv.h"
#include "Object3D.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <vector>
#include <string>
#include <iostream>

namespace py = pybind11;

class Space3D : public BaseEnv
{
private:
    double dt;
    double eps;

public:
    std::vector<std::shared_ptr<Object3D>> objects;

    Space3D(double dt, double eps) : dt(dt), eps(eps) {}

    py::object reset()
    {
        for (auto& object : objects)
        {
            object->reset();
        }
        return to_dict();
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

    py::object step(const py::object& actions)
    {
        py::dict result, info;

        for (auto& object : objects)
        {
            result[py::str(object->name)] = object->step(actions[py::str(object->name)]);
        }

        return result;
    }
};