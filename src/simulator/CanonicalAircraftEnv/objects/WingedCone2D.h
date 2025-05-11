#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
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

    virtual void reset() override
    {
        Aircraft3D::reset();
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
        double CM2 = ang_vel_b(1) * c * (-6.796 * alpha * alpha + 0.3015 * alpha - 0.2289) / (2 * V);
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
        
        force_vec c_force = {T - D * cos(alpha) * cos(beta) - N * sin(beta) * cos(alpha) + L * sin(alpha) - m * g * sin(theta),
                             -D * sin(beta) + N * cos(beta) + m * g * cos(theta) * sin(gamma),
                             -D * cos(beta) * sin(alpha) - N * sin(beta) * sin(alpha) - L * cos(alpha) + m * g * cos(theta) * cos(gamma),
                             M[0], M[1], M[2]}; // 力和力矩
        kinematics_step(c_force); // 更新状态

        h = pos[1];

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;

        return to_dict();
    }
};
