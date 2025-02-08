#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

class SimulatorEnv {
public:
    py::dict process_input(py::dict input_dict) {
        if (input_dict.contains("value")) {
            try {
                int value = input_dict["value"].cast<int>();
                input_dict["value"] = value + 1;

            } catch (const std::exception &e) {
                input_dict["error"] = std::string("failed to process input: ") + e.what();
            }

        } else {
            input_dict["error"] = "missing 'value' key in input dictionary";
        }


        return input_dict;
    }
};

PYBIND11_MODULE(simulator, m) {
    py::class_<SimulatorEnv>(m, "SimulatorEnv")
        .def(py::init<>())
        .def("process_input", &SimulatorEnv::process_input, "process input dictionary and return updated dictionary");
}