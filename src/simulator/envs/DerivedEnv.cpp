#include "BaseEnv.h"
#include "BaseEnv.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class DerivedEnv : public BaseEnv {
public:
    DerivedEnv() {
        inner_state = 0;
    }

    ~DerivedEnv() {}

    py::object reset() override {
        py::dict result;
        inner_state = 0;
        return result;
    }

    py::object step(const py::object& input_dict) override {
        inner_state ++;
        py::dict result = input_dict;
        if (input_dict.contains("value"))
        {
            try {
                int value = input_dict["value"].cast<int>();
                result["value"] = value + 1;
            } catch (const std::exception &e) {
                result["error"] = std::string("failed to convert value to int: ") + e.what();
            }
        }
        else
        {
            result["error"] = "error: input_dict is not a dict";
        }
        
        result["inner_state"] = inner_state;
        return result;
    }

private:
    int inner_state = 0;

};

PYBIND11_MODULE(DerivedEnv, m) {
    py::class_<DerivedEnv, BaseEnv>(m, "DerivedEnv")
        .def(py::init<>())
        .def("reset", &DerivedEnv::reset)
        .def("step", &DerivedEnv::step);
}
