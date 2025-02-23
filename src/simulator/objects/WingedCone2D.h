#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "y_atmosphere.h"
#include "Object3D.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

class WingedCone2D : public Object3D
{
public:
    inline static const double S = 3603; // 参考面积
    inline static const double c = 80; // 特征长度
    inline static const double m0 = 9375; // 质量
    inline static const double Iyy = 7 * 10e6; // 俯仰转动惯量

    double V; // 速度
    double h; // 高度
    double m; // 质量
    double q; // 动压

    double Tem; // 温度
    double Pres; // 压力
    double Rho; // 空气密度
    double a; // 声速
    double g; // 重力加速度

    double D; // 阻力
    double L; // 升力
    double T; // 推力
    double M; // 俯仰力矩

    double delta_e; // 升降舵偏角
    
    WingedCone2D()
    {
        // Default constructor implementation (if needed)
    }

    WingedCone2D(py::dict input_dict) : Object3D(input_dict)
    {
        V = std::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
        h = pos[1];
        m = m0;

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;
    }

    double _D()
    {
        double CD = 0.645 * alpha * alpha + 0.0043378 * alpha + 0.003772;
        D = q * S * CD;
        return D;
    }

    double _L()
    {
        double CL = 0.6203 * alpha + 2.4 * sin(0.08 * alpha);
        L = q * S * CL;
        return L;
    }

    double WingedCone2D::_T()
    {
        T = 4.959e3;
        return T;
    }

    double WingedCone2D::_M()
    {
        double CM1 = -0.035 * alpha * alpha + 0.036617 * alpha + 5.3261e-6;
        double CM2 = ang_vel[2] * c * (-6.796 * alpha * alpha + 0.3015 * alpha - 0.2289) / (2 * V);
        double CM3 = 0.0292 * (delta_e - alpha);
        M = q * S * c * (CM1 + CM2 + CM3);
        return M;
    }

    py::object WingedCone2D::step(py::dict input_dict)
    {
        py::dict obs;
        delta_e = input_dict["delta_e"].cast<double>();
        
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
