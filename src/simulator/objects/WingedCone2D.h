#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <y_atmosphere.h>
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
    
    WingedCone2D();
    WingedCone2D(py::dict input_dict);

    double _D();
    double _L();
    virtual double _T();
    double _M();
    virtual py::object step(py::dict input_dict);
};
