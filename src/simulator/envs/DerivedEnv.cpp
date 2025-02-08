#include "BaseEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

class Env : public BaseEnv {
public:
    Env() : state(0) {}

    ~Env() {}

    py::object reset() override {
        state = 0;
        py::dict result;
        result["state"] = state;
        return result;
    }


    py::object step(const py::object& input_dict) override {
        py::dict result;
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
        
        result["state"] = state;
        return result;
    }

private:
    int state;
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
