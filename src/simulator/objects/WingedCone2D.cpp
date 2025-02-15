#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <y_atmosphere.h>
#include <Object3D.cpp>
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

class WingedCone2D : public Object3D
{
public:
    WingedCone2D();
    WingedCone2D(py::dict input_dict) : Object3D(input_dict) {};
};

