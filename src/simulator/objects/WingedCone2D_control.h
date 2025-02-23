#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "WingedCone2D.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

class WingedCone2D_control : public WingedCone2D
{
public:
    // 俯仰角增稳过载驾驶仪控制参数
    double Kiz;   // 积分增益
    double Kwz;   // 角速度增益
    double Kaz;   // 增稳回路增益
    double Kpz;   // 比例增益

    double eNy; // 过载跟踪误差
    double i_eNy; // 过载积分项
    double p_eNy; // 过载比例项

    double i_eSAC; // 增稳回路积分项

    double Kp_V, Ki_V, Kd_V; // 速度控制参数

    double i_V; // 速度积分项
    double d_eV; // 速度微分项
    double eV_prev; // 速度误差前值

    WingedCone2D_control();

    double _T()
    {
        T = 4.959e3;
        return T;
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

    double Ny_controller(double Nyc, double Ny, double wz, double dt)
    {
        // 过载跟踪误差
        eNy = Nyc - Ny;

        // PI校正环节
        i_eNy += eNy * dt;
        p_eNy = eNy;

        double pi_eNy = Kiz * i_eNy + Kpz * p_eNy;

        // 增稳回路
        double eSAC = pi_eNy - Kaz * wz;
        i_eSAC += eSAC * dt;

        // 阻尼回路
        double eDamp = i_eSAC - Kwz * wz;

        return eDamp;
    }

    py::object step(py::dict input_dict)
    {
        py::dict obs;
        double Nyc = input_dict["Nyc"].cast<double>();
        double Vc = input_dict["Vc"].cast<double>();
        double dt = input_dict["dt"].cast<double>();

        double Ny = acc[1] / g;
        double wz = ang_acc[2];

        delta_e = Ny_controller(Nyc, Ny, wz, dt);
        delta_e = std::clamp(delta_e, -25 / 57.3, 25 / 57.3);

        double Phi = V_controller(Vc, V, dt);
        
        // 计算气动力
        D = _D();
        L = _L();
        T = _T();
        M = _M();

        // 计算加速度
        acc[0] = (T * cos(alpha) - D) / m;
        acc[1] = (L + T * sin(alpha)) / m;
        acc[2] = 0;

        // 计算角加速度
        ang_acc[2] = M / Iyy;
        ang_acc[1] = 0;
        ang_acc[2] = 0;

        return obs;
    }
};
