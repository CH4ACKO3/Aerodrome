#include "BaseEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class Env : public BaseEnv {
public:
    Env() {
        inner_state = 0;
    }

    ~Env() {}

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
    py::class_<BaseEnv>(m, "BaseEnv")
        .def("reset", &BaseEnv::reset)
        .def("step", &BaseEnv::step);

    py::class_<Env, BaseEnv>(m, "Env")
        .def(py::init<>())
        .def("reset", &Env::reset)
        .def("step", &Env::step);
}
