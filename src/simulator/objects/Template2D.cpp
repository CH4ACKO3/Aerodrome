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

class Template2D : public Object3D
{
public:
    static const double S; // 参考面积
    static const double c; // 特征长度
    static const double m0; // 质量
    static const double Iyy; // 俯仰转动惯量

    double V; // 速度
    double h; // 高度
    double m; // 质量
    double q; // 动压

    double Rho; // 空气密度
    double Tem; // 温度
    double Pres; // 压力
    double a; // 声速
    double g; // 重力加速度

    double D; // 阻力
    double L; // 升力
    double T; // 推力
    double M; // 俯仰力矩
    
    Template2D();
    Template2D(py::dict input_dict) : Object3D(input_dict)
    {
        S = input_dict["S"];
        c = input_dict["c"];
        m0 = input_dict["m0"];
        Iyy = input_dict["Iyy"];

        V = input_dict["V"];
        h = input_dict["h"];
        m = input_dict["m"];

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;
    };

    double D() {};
    double L() {};
    double M() {};

    void step(py::dict input_dict)
    {
        // 计算气动力
    };
};

