#pragma once

#include "BaseEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <string>

namespace py = pybind11;

class MinimalEnv : public BaseEnv { // inherit from BaseEnv
public:
    MinimalEnv() : cpp_state(0) {} // initialize cpp_state to 0

    ~MinimalEnv() {} // destructor

    py::object reset() override // override the reset method defined in BaseEnv
    {
        py::dict result; // create a dictionary to store the result to be returned
        cpp_state = 0; // reset cpp_state to 0
        result["cpp_state"] = cpp_state;
        result["info"] = "";
        return result;
    }

    py::object step(const py::object& action) override // step method receives an action from the python side
    {
        // In each step, the environment adds the value of action["value"] to its cpp_state
        py::dict result;
        if (action.contains("value"))
        {
            try {
                int value = action["value"].cast<int>(); // convert the value of action["value"] to an integer
                cpp_state += value; // add the value to cpp_state
            } catch (const std::exception &e) {
                result["info"] = std::string("failed to convert value to int: ") + e.what(); // if the value is not an integer, add an error message to the result
            }
        }
        else
        {
            result["info"] = "error: action must contain 'value'";
        }
        
        result["cpp_state"] = cpp_state;
        if (!result.contains("info"))
        {
            // if the info key is not in the result, add an empty string to it, to avoid key error
            result["info"] = "";
        }
        return result;
    }

private:
    int cpp_state = 0; // private member variable to store the state of the environment
};