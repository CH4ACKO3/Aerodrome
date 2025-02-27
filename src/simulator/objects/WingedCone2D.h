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
        double dt = action["dt"].cast<double>();

        delta_e = action["delta_e"].cast<double>();
        
        // 计算气动力
        _D();
        _L();
        _T();
        _M();

        if (integrator == "euler")
        {
            *this = *this + this->d() * dt;
        }
        else if (integrator == "midpoint")
        {
            auto temp1 = *this + this->d() * (0.5 * dt);
            auto k1 = temp1.d();
            *this = *this + k1 * dt;
        }
        else if (integrator == "rk23")
        {
            auto k1 = this->d();
            auto temp1 = *this + k1 * (0.5 * dt);
            auto k2 = temp1.d();
            auto temp2 = *this + k2 * (0.5 * dt);
            auto k3 = temp2.d();
            *this = *this + (k1 + k2 * 2 + k3) * (dt / 4);
        }
        else if (integrator == "rk45")
        {
            auto k1 = this->d();
            auto temp1 = *this + k1 * (0.5 * dt);
            auto k2 = temp1.d();
            auto temp2 = *this + k2 * (0.5 * dt);
            auto k3 = temp2.d();
            auto temp3 = *this + k3 * dt;
            auto k4 = temp3.d();
            *this = *this + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6);
        }

        beta = cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma);
        alpha = (cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta);
        gamma_v = (cos(alpha) * sin(beta) * sin(theta) - sin(alpha) * sin(beta) * cos(gamma) * cos(theta) + cos(beta) * sin(gamma) * cos(theta)) / cos(theta_v);

        vel[0] = V * cos(theta_v) * cos(phi_v);
        vel[1] = V * sin(theta_v);
        vel[2] = -V * cos(theta_v) * sin(phi_v);
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
