#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <y_atmosphere.h>
#include <Object3D.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

class Template2D : public Object3D
{
public:
    inline static const double S = 0.0; // 参考面积
    inline static const double c = 0.0; // 特征长度
    inline static const double m0 = 0.0; // 质量
    inline static const double Iyy = 0.0; // 俯仰转动惯量

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
        V = input_dict["V"].cast<double>();
        h = input_dict["h"].cast<double>();
        m = input_dict["m"].cast<double>();

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;
    };

    double _D() {};
    double _L() {};
    double _T() {};
    double _M() {};

    py::object step(py::dict input_dict)
    {
        // 计算气动力
        return py::none();
    };
};

