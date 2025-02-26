#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Aircraft3D.h"
#include "y_atmosphere.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

class WingedCone2D : public Aircraft3D
{
public:
    double delta_e; // 升降舵偏角
    
    WingedCone2D() {}

    WingedCone2D(py::dict input_dict) : Aircraft3D(input_dict)
    {
        delta_e = 0.0;        
    }

    void _D()
    {
        double CD = 0.645 * alpha * alpha + 0.0043378 * alpha + 0.003772;
        D = q * S * CD;
    }

    void _L()
    {
        double CL = 0.6203 * alpha + 2.4 * sin(0.08 * alpha);
        L = q * S * CL;
    }

    virtual void _T()
    {
        T = 4.959e3;
    }

    void _M()
    {
        double CM1 = -0.035 * alpha * alpha + 0.036617 * alpha + 5.3261e-6;
        double CM2 = ang_vel[2] * c * (-6.796 * alpha * alpha + 0.3015 * alpha - 0.2289) / (2 * V);
        double CM3 = 0.0292 * (delta_e - alpha);
        M[2] = q * S * c * (CM1 + CM2 + CM3);
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = Aircraft3D::to_dict();
        output_dict["delta_e"] = delta_e;
        return output_dict;
    }

    virtual py::object step(py::dict action) override
    {
        delta_e = action["delta_e"].cast<double>();
        
        // 计算气动力
        _D();
        _L();
        _T();
        _M();

        kinematics_step(action["dt"].cast<double>());
        update(action["dt"].cast<double>());

        return to_dict();
    }

    virtual WingedCone2D* clone() const override
    {
        return new WingedCone2D(*this);
    }
};
