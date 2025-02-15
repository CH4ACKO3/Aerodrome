#include "BaseEnv.h"
#include "BaseEnv.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

class Object3D {
public:
    std::string name;

    // 位置坐标（地面系）
    std::array<double, 3> pos;

    // 速度向量（地面系）
    std::array<double, 3> vel;

    // 加速度向量（地面系）
    std::array<double, 3> acc;

    // 角速度向量（弹体系）
    std::array<double, 3> ang_vel;

    // 角加速度向量（弹体系）
    std::array<double, 3> ang_acc;

    // 描述刚体姿态的八个角度
    double theta;    // 俯仰角
    double phi;      // 偏航角
    double gamma;    // 倾斜角
    double theta_v;  // 速度倾角
    double phi_v;    // 速度偏角
    double alpha;    // 攻角
    double beta;     // 侧滑角
    double gamma_v;  // 速度倾斜角

    Object3D() 
        : name(""), pos{0.0, 0.0, 0.0}, vel{0.0, 0.0, 0.0}, acc{0.0, 0.0, 0.0},
          ang_vel{0.0, 0.0, 0.0}, ang_acc{0.0, 0.0, 0.0},
          theta(0.0), phi(0.0), gamma(0.0), theta_v(0.0), phi_v(0.0),
          alpha(0.0), beta(0.0), gamma_v(0.0) {}
    
    Object3D(py::dict input_dict)
    {
        name = input_dict["name"].cast<std::string>();
        pos = input_dict["pos"].cast<std::array<double, 3>>();
        vel = input_dict["vel"].cast<std::array<double, 3>>();
        acc = input_dict["acc"].cast<std::array<double, 3>>();
        ang_vel = input_dict["ang_vel"].cast<std::array<double, 3>>();
        ang_acc = input_dict["ang_acc"].cast<std::array<double, 3>>();
        theta = input_dict["theta"].cast<double>();
        phi = input_dict["phi"].cast<double>();
        gamma = input_dict["gamma"].cast<double>();
        theta_v = input_dict["theta_v"].cast<double>();
        phi_v = input_dict["phi_v"].cast<double>();
        gamma_v = input_dict["gamma_v"].cast<double>();
        alpha = input_dict["alpha"].cast<double>();
        beta = input_dict["beta"].cast<double>();
    }

    py::dict to_dict()
    {
        py::dict output_dict;
        output_dict["pos"] = pos;
        output_dict["vel"] = vel;
        output_dict["acc"] = acc;
        output_dict["ang_vel"] = ang_vel;
        output_dict["ang_acc"] = ang_acc;
        output_dict["theta"] = theta;
        output_dict["phi"] = phi;
        output_dict["gamma"] = gamma;
        output_dict["theta_v"] = theta_v;
        output_dict["phi_v"] = phi_v;
        output_dict["alpha"] = alpha;
        output_dict["beta"] = beta;
        output_dict["gamma_v"] = gamma_v;
        output_dict["name"] = name;
        return output_dict;
    }

    Object3D operator+(const Object3D& other) const
    {
        Object3D result;
        result.name = this->name;
        for (size_t i = 0; i < 3; ++i)
        {
            result.pos[i] = this->pos[i] + other.pos[i];
            result.vel[i] = this->vel[i] + other.vel[i];
            result.acc[i] = this->acc[i] + other.acc[i];
            result.ang_vel[i] = this->ang_vel[i] + other.ang_vel[i];
            result.ang_acc[i] = this->ang_acc[i] + other.ang_acc[i];
        }
        result.theta = this->theta + other.theta;
        result.phi = this->phi + other.phi;
        result.gamma = this->gamma + other.gamma;
        result.theta_v = this->theta_v + other.theta_v;
        result.phi_v = this->phi_v + other.phi_v;
        result.alpha = this->alpha + other.alpha;
        result.beta = this->beta + other.beta;
        result.gamma_v = this->gamma_v + other.gamma_v;
        return result;
    }

    Object3D operator*(const double& other) const
    {
        Object3D result;
        result.name = this->name;
        for (size_t i = 0; i < 3; ++i)
        {
            result.pos[i] = this->pos[i] * other;   
            result.vel[i] = this->vel[i] * other;
            result.acc[i] = this->acc[i] * other;
            result.ang_vel[i] = this->ang_vel[i] * other;
            result.ang_acc[i] = this->ang_acc[i] * other;
        }
        result.theta = this->theta * other;
        result.phi = this->phi * other; 
        result.gamma = this->gamma * other;
        result.theta_v = this->theta_v * other;
        result.phi_v = this->phi_v * other;
        result.alpha = this->alpha * other;
        result.beta = this->beta * other;
        result.gamma_v = this->gamma_v * other;
        return result;  
    }

    Object3D operator/(const double& other) const
    {
        Object3D result;
        result.name = this->name;
        for (size_t i = 0; i < 3; ++i)
        {
            result.pos[i] = this->pos[i] / other;   
            result.vel[i] = this->vel[i] / other;
            result.acc[i] = this->acc[i] / other;
            result.ang_vel[i] = this->ang_vel[i] / other;
            result.ang_acc[i] = this->ang_acc[i] / other;
        }
        result.theta = this->theta / other; 
        result.phi = this->phi / other; 
        result.gamma = this->gamma / other;
        result.theta_v = this->theta_v / other;
        result.phi_v = this->phi_v / other;
        result.alpha = this->alpha / other;
        result.beta = this->beta / other;
        result.gamma_v = this->gamma_v / other; 
        return result;
    }
};

class Space3D : public BaseEnv
{
protected:
    std::vector<std::shared_ptr<Object3D>> objects;
    double tau; // 时间步长
    double dt; // 积分步长
    double eps; // 自洽性容差
    std::string integrator; // 积分器类型
    int integrate_steps; // 每步积分次数

    std::shared_ptr<Object3D> d(std::shared_ptr<Object3D> object)
    {
        auto next_object = std::make_shared<Object3D>();
        auto derivative = std::make_shared<Object3D>();

        next_object->theta = object->theta + object->ang_vel[2] * dt;
        next_object->phi = object->phi + object->ang_vel[1] * dt;
        next_object->gamma = object->gamma + object->ang_vel[0] * dt;

        next_object->theta_v = atan2(object->vel[1], sqrt(object->vel[0] * object->vel[0] + object->vel[2] * object->vel[2]));
        next_object->phi_v = atan2(-object->vel[2], object->vel[0]);

        double sin_beta = cos(next_object->theta_v) *
                            (cos(next_object->gamma) * sin(next_object->phi - next_object->phi_v) +
                            sin(next_object->theta) * sin(next_object->gamma) * cos(next_object->phi - next_object->phi_v)) -
                            sin(next_object->theta_v) * cos(next_object->theta) * sin(next_object->gamma);
        double cos_beta = sqrt(1 - sin_beta * sin_beta);

        double sin_alpha = (cos(next_object->theta_v) *
                            (sin(next_object->theta) * cos(next_object->gamma) * cos(next_object->phi - next_object->phi_v) - sin(next_object->gamma) * sin(next_object->phi - next_object->phi_v)) -
                            sin(next_object->theta_v) * cos(next_object->theta) * cos(next_object->gamma)) / cos_beta;
        double cos_alpha = sqrt(1 - sin_alpha * sin_alpha);

        double sin_gamma_v = (cos_alpha * sin_beta * sin(next_object->theta) - 
                                sin_alpha * sin_beta * cos(next_object->gamma) * cos(next_object->theta) +
                                cos_beta * sin(next_object->gamma) * cos(next_object->theta)) / cos(next_object->theta_v);
        
        next_object->alpha = asin(sin_alpha);
        next_object->beta = asin(sin_beta);
        next_object->gamma_v = asin(sin_gamma_v);

        for (size_t i = 0; i < 3; ++i)
        {
            derivative->pos[i] = object->vel[i];
            derivative->vel[i] = object->acc[i];
            derivative->ang_vel[i] = object->ang_acc[i];
        }

        derivative->theta = object->ang_vel[2];
        derivative->phi = object->ang_vel[1];
        derivative->gamma = object->ang_vel[0];

        derivative->theta_v = (next_object->theta_v - object->theta_v) / dt;
        derivative->phi_v = (next_object->phi_v - object->phi_v) / dt;
        derivative->alpha = (next_object->alpha - object->alpha) / dt;
        derivative->beta = (next_object->beta - object->beta) / dt;
        derivative->gamma_v = (next_object->gamma_v - object->gamma_v) / dt;

        return derivative;
    }

public:
    Space3D(double tau = 1e-2, double eps = 1e-6, std::string integrator = "euler", int integrate_steps = 1) : tau(tau), eps(eps), integrator(integrator), integrate_steps(integrate_steps)
    {
        dt = tau / integrate_steps;
    }

    void add_object(std::shared_ptr<Object3D> object)
    {
        objects.push_back(object);
    }

    bool check_consistency()
    {
        return true;
    }

    py::dict to_dict()
    {
        py::dict output_dict;
        for (auto& object : objects)
        {
            output_dict[py::str(object->name)] = object->to_dict();
        }
        return output_dict;
    }

    virtual py::object reset() override
    {   
        return py::none();
    }

    virtual py::object step(const py::object& action) override
    {
        return py::none();
    }

    void kinematics_step()
    {
        for (int i = 0; i < integrate_steps; ++i)
        {
            for (auto& object : objects)
            {
                if (integrator == "euler")
                {
                    *object = *object + *d(object) * dt;
                }
                else if (integrator == "midpoint")
                {
                    auto temp1 = std::make_shared<Object3D>(*object + *d(object) * dt * 0.5);
                    *object = *object + *d(temp1) * dt;
                }
                else if (integrator == "rk23")
                {
                    auto k1 = d(object);
                    auto temp1 = std::make_shared<Object3D>(*object + *k1 * dt * 0.5);
                    auto k2 = d(temp1);
                    auto temp2 = std::make_shared<Object3D>(*object + *k2 * dt * 0.5);
                    auto k3 = d(temp2);
                    *object = *object + (*k1 + *k2 * 2 + *k3) * dt / 4;
                }
                else if (integrator == "rk45")
                {
                    auto k1 = d(object);
                    auto temp1 = std::make_shared<Object3D>(*object + *k1 * dt * 0.5);
                    auto k2 = d(temp1);
                    auto temp2 = std::make_shared<Object3D>(*object + *k2 * dt * 0.5);
                    auto k3 = d(temp2);
                    auto temp3 = std::make_shared<Object3D>(*object + *k3 * dt);
                    auto k4 = d(temp3);
                    *object = *object + (*k1 + *k2 * 2 + *k3 * 2 + *k4) * dt / 6;
                }
            }
        }
    }
};

PYBIND11_MODULE(Space3D, m)
{
    py::class_<Object3D, std::shared_ptr<Object3D>>(m, "Object3D")
        .def(py::init<>())
        .def(py::init<py::dict>())
        .def("to_dict", &Object3D::to_dict)
        .def("__add__", &Object3D::operator+)
        .def("__mul__", &Object3D::operator*)
        .def("__truediv__", &Object3D::operator/);

    py::class_<Space3D, BaseEnv>(m, "Space3D")
        .def(py::init<double, double, std::string, int>(), pybind11::arg("tau") = 0.01, pybind11::arg("eps") = 0.000001, pybind11::arg("integrator") = "euler", pybind11::arg("integrate_steps") = 1)
        .def("add_object", &Space3D::add_object)
        .def("check_consistency", &Space3D::check_consistency)
        .def("to_dict", &Space3D::to_dict)
        .def("kinematics_step", &Space3D::kinematics_step);
}
