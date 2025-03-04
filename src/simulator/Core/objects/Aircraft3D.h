#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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
    double S; // 参考面积
    double c; // 特征长度
    double m; // 质量

    double h; // 高度
    double q; // 动压

    double Rho; // 空气密度
    double Tem; // 温度
    double Pres; // 压力
    double a; // 声速
    double g; // 重力加速度

    double L; // 升力
    double D; // 阻力
    double N; // 侧力
    double T; // 推力
    std::array<double, 3> M; // 力矩

    Aircraft3D() {}

    Aircraft3D(py::dict input_dict) : Object3D(input_dict)
    {
        V = sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
        h = pos[1];
        S = input_dict["S"].cast<double>();
        c = input_dict["c"].cast<double>();
        m = input_dict["m"].cast<double>();

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
        *this = Aircraft3D(initial_state);
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = Object3D::to_dict();
        output_dict["S"] = S;
        output_dict["c"] = c;
        output_dict["m"] = m;
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
        double dt = action["dt"].cast<double>();
        
        L = action["L"].cast<double>();
        D = action["D"].cast<double>();
        N = action["N"].cast<double>();
        T = action["T"].cast<double>();
        M = action["M"].cast<std::array<double, 3>>();

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

    virtual Object3D d() override
    {   
        auto derivative = *this;
        
        derivative.V = (T * cos(alpha) * cos(beta) - D - m * g * sin(theta_v)) / m;
        derivative.theta_v = (T * (sin(alpha) * cos(gamma_v) - cos(alpha) * sin(beta) * sin(gamma_v))
                                + L * cos(gamma_v) - N * sin(gamma_v) - m * g * cos(theta_v)) / (m * V);
        derivative.phi_v = -(T * (sin(alpha) * sin(gamma_v) - cos(alpha) * sin(beta) * cos(gamma_v))
                            + L * sin(gamma_v) + N * cos(gamma_v)) / (m * V * cos(theta_v));

        derivative.ang_vel[0] = (M[0] - (J[2] - J[1]) * ang_vel[1] * ang_vel[2]) / J[0];
        derivative.ang_vel[1] = (M[1] - (J[0] - J[2]) * ang_vel[2] * ang_vel[0]) / J[1];
        derivative.ang_vel[2] = (M[2] - (J[1] - J[0]) * ang_vel[0] * ang_vel[1]) / J[2];

        derivative.theta = ang_vel[1] * sin(gamma) + ang_vel[2] * cos(gamma);
        derivative.phi = (ang_vel[1] * cos(gamma) - ang_vel[2] * sin(gamma)) / cos(theta);
        derivative.gamma = ang_vel[0] * - tan(theta) * (ang_vel[1] * cos(gamma) - ang_vel[2] * sin(gamma));

        derivative.pos[0] = V * cos(theta_v) * cos(phi_v);
        derivative.pos[1] = V * sin(theta_v);
        derivative.pos[2] = -V * cos(theta_v) * sin(phi_v);

        return derivative;
    }
};