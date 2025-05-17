#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Object3D.h"
#include "y_atmosphere.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

class Aircraft3D : public Object3D {
public:
    double S; // Reference area
    double c; // Characteristic length

    double h; // Height/Altitude
    double q; // Dynamic pressure

    double Rho; // Air density
    double Tem; // Temperature
    double Pres; // Pressure
    double a; // Speed of sound
    double g; // Gravitational acceleration

    double L; // Lift, defined in wind
    double D; // Drag
    double N; // Side force
    double T; // Thrust
    std::array<double, 3> M; // Moment

    Aircraft3D() {}

    Aircraft3D(py::dict input_dict) : Object3D(input_dict)
    {
        h = -pos[2];
        S = input_dict["S"].cast<double>();
        c = input_dict["c"].cast<double>();

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;

        L = 0.0;
        D = 0.0;
        N = 0.0;
        T = 0.0;
        M = {0.0, 0.0, 0.0};
    }

    virtual void reset() override
    {
        Object3D::reset();
        
        h = -pos[2];

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;

        L = 0.0;
        D = 0.0;
        N = 0.0;
        T = 0.0;
        M = {0.0, 0.0, 0.0};
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = Object3D::to_dict();
        output_dict["S"] = S;
        output_dict["c"] = c;
        output_dict["V"] = V;
        output_dict["h"] = h;
        output_dict["q"] = q;
        output_dict["Rho"] = Rho;
        output_dict["Tem"] = Tem;
        output_dict["Pres"] = Pres;
        output_dict["a"] = a;
        output_dict["g"] = g;
        output_dict["L"] = L;
        output_dict["D"] = D;
        output_dict["N"] = N;
        output_dict["T"] = T;
        output_dict["M"] = M;
        return output_dict;
    }

    virtual py::object step(py::dict action) override
    {
        L = action["L"].cast<double>();
        D = action["D"].cast<double>();
        N = action["N"].cast<double>();
        T = action["T"].cast<double>();
        M = action["M"].cast<std::array<double, 3>>();
        
        force_vec c_force = {T - D * cos(alpha) * cos(beta) - N * sin(beta) * cos(alpha) + L * sin(alpha) - m * g * sin(theta),
                             -D * sin(beta) + N * cos(beta) + m * g * cos(theta) * sin(gamma),
                             -D * cos(beta) * sin(alpha) - N * sin(beta) * sin(alpha) - L * cos(alpha) + m * g * cos(theta) * cos(gamma),
                             M[0], M[1], M[2]}; // 力和力矩
        kinematics_step(c_force); // 更新状态

        h = -pos[2];

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;

        return to_dict();
    }
};