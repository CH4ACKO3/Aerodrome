#pragma once

#include "BaseEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class MinimalEnv : public BaseEnv {
public:
    MinimalEnv() : cpp_state(0) {}

    ~MinimalEnv() {}

    py::object reset() override
    {
        py::dict result;
        cpp_state = 0;
        result["cpp_state"] = cpp_state;
        result["info"] = "";
        return result;
    }

    py::object step(const py::object& action) override
    {
        py::dict result;
        if (action.contains("value"))
        {
            try {
                int value = action["value"].cast<int>();
                cpp_state += value;
            } catch (const std::exception &e) {
                result["info"] = std::string("failed to convert value to int: ") + e.what();
            }
        }
        else
        {
            result["info"] = "error: action must contain 'value'";
        }
        
        result["cpp_state"] = cpp_state;
        if (!result.contains("info"))
        {
            result["info"] = "";
        }
        return result;
    }

private:
    int cpp_state = 0;
};