#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "WingedCone2D.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

class WingedCone2D_RL : public WingedCone2D
{
public:
    // 俯仰角增稳过载驾驶仪控制参数
    double Kiz;   // 积分增益
    double Kwz;   // 角速度增益
    double Kaz;   // 增稳回路增益
    double Kpz;   // 比例增益

    double eNy; // 过载跟踪误差
    double eNy_prev; // 过载误差前值
    double i_eNy; // 过载积分项
    double d_eNy; // 过载比例项

    double i_eSAC; // 增稳回路积分项

    double Kp_V, Ki_V, Kd_V; // 速度控制参数

    double i_V; // 速度积分项
    double d_eV; // 速度微分项
    double eV_prev; // 速度误差前值

    double Ny; // 当前过载
    double wz; // 当前滚转角速度

    WingedCone2D_RL() {}

    WingedCone2D_RL(py::dict input_dict) : WingedCone2D(input_dict)
    {
        Kiz = input_dict["Kiz"].cast<double>();
        Kwz = input_dict["Kwz"].cast<double>();
        Kaz = input_dict["Kaz"].cast<double>();
        Kpz = input_dict["Kpz"].cast<double>();

        Kp_V = input_dict["Kp_V"].cast<double>();
        Ki_V = input_dict["Ki_V"].cast<double>();
        Kd_V = input_dict["Kd_V"].cast<double>();

        eNy = 0;
        eNy_prev = 0;
        i_eNy = 0;
        d_eNy = 0;
        i_eSAC = 0;
        i_V = 0;
        d_eV = 0;
        eV_prev = 0;

        _D();
        _L();
        _T();
        _M();

        Ny = (T * (sin(alpha) * cos(gamma_v) - cos(alpha) * sin(beta) * sin(gamma_v))
                                + L * cos(gamma_v) - N * sin(gamma_v) - m * g * cos(theta_v)) / (m * g);
        wz = ang_vel[2];
    }

    virtual void reset() override
    {
        WingedCone2D::reset();
        eNy = 0;
        eNy_prev = 0;
        i_eNy = 0;
        d_eNy = 0;
        i_eSAC = 0;
        i_V = 0;
        d_eV = 0;
        eV_prev = 0;
    }

    double V_controller(double Vc, double V, double dt)
    {
        // 速度跟踪误差
        double eV = Vc - V;
        i_V += eV * dt;
        d_eV = (eV - eV_prev) / dt;
        eV_prev = eV;

        double u1a = Kp_V * eV + Ki_V * i_V + Kd_V * d_eV;
        if (u1a < 0) 
        {
            u1a = 0;
        }

        return u1a;
    }

    double Ny_controller(double nn_control, double wz, double dt)
    {
        // 增稳回路，使用神经网络输出代替PI环节
        double eSAC = nn_control - Kaz * wz;
        i_eSAC += eSAC * dt;

        // 阻尼回路
        double eDamp = i_eSAC - Kwz * wz;

        return eDamp;
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = WingedCone2D::to_dict();
        output_dict["Ny"] = Ny;
        output_dict["eNy"] = eNy;
        output_dict["i_eNy"] = i_eNy;
        output_dict["d_eNy"] = d_eNy;

        return output_dict;
    }

    virtual py::object step(py::dict action) override
    {
        double Nyc = action["Nyc"].cast<double>();
        double Vc = action["Vc"].cast<double>();
        double nn_control = action["nn_control"].cast<double>();

        delta_e = Ny_controller(nn_control, wz, dt*0.1);
        delta_e = std::clamp(delta_e, -25 / 57.3, 25 / 57.3);

        double Phi = V_controller(Vc, V, dt*0.1);

        // 计算气动力
        _D();
        _L();
        _T();
        _M();
        
        force_vec c_force = {T * cos(alpha) * cos(beta) - D - m * g * sin(theta),
                             T * (sin(alpha) * cos(gamma_v) + cos(alpha) * sin(beta) * sin(gamma_v)) + L * cos(gamma_v) - N * sin(gamma_v) - m * g * cos(theta),
                             T * (sin(alpha) * sin(gamma_v) - cos(alpha) * sin(beta) * cos(gamma_v)) + L * sin(gamma_v) + N * cos(gamma_v),
                             M[0], M[1], M[2]}; // 力和力矩
        kinematics_step(c_force); // 更新状态

        h = pos[1];

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;
        
        Ny = (T * (sin(alpha) * cos(gamma_v) - cos(alpha) * sin(beta) * sin(gamma_v))
                                + L * cos(gamma_v) - N * sin(gamma_v) - m * g * cos(theta_v)) / (m * g);
        wz = ang_vel[2];
        
        return to_dict();
    }
};
