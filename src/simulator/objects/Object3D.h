#pragma once

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
    std::string integrator;

    // 位置坐标（地面系）
    std::array<double, 3> pos;

    // 速度向量（地面系）
    std::array<double, 3> vel;

    // 角速度向量（弹体系）
    std::array<double, 3> ang_vel;

    // 转动惯量
    std::array<double, 3> J;

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
        : name(""), integrator("euler"), pos{0.0, 0.0, 0.0}, vel{0.0, 0.0, 0.0}, ang_vel{0.0, 0.0, 0.0}, J{0.0, 0.0, 0.0},
          theta(0.0), phi(0.0), gamma(0.0), theta_v(0.0), phi_v(0.0),
            alpha(0.0), beta(0.0), gamma_v(0.0) {}
        
    Object3D(py::dict input_dict)
    {
        name = input_dict["name"].cast<std::string>();
        integrator = input_dict["integrator"].cast<std::string>();
        pos = input_dict["pos"].cast<std::array<double, 3>>();
        vel = input_dict["vel"].cast<std::array<double, 3>>();
        ang_vel = input_dict["ang_vel"].cast<std::array<double, 3>>();
        J = input_dict["J"].cast<std::array<double, 3>>();
        theta = input_dict["theta"].cast<double>();
        phi = input_dict["phi"].cast<double>();
        gamma = input_dict["gamma"].cast<double>();
        theta_v = input_dict["theta_v"].cast<double>();
        phi_v = input_dict["phi_v"].cast<double>();
        gamma_v = input_dict["gamma_v"].cast<double>();
        alpha = input_dict["alpha"].cast<double>();
        beta = input_dict["beta"].cast<double>();
    }

    virtual py::dict to_dict()
    {
        py::dict output_dict;
        output_dict["pos"] = pos;
        output_dict["vel"] = vel;
        output_dict["ang_vel"] = ang_vel;
        output_dict["J"] = J;
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

    virtual py::object step(py::dict action)
    {
        pos = action["pos"].cast<std::array<double, 3>>();
        vel = action["vel"].cast<std::array<double, 3>>();
        ang_vel = action["ang_vel"].cast<std::array<double, 3>>();
        J = action["J"].cast<std::array<double, 3>>();
        theta = action["theta"].cast<double>();
        phi = action["phi"].cast<double>();
        gamma = action["gamma"].cast<double>();
        theta_v = action["theta_v"].cast<double>();
        phi_v = action["phi_v"].cast<double>();
        gamma_v = action["gamma_v"].cast<double>();
        alpha = action["alpha"].cast<double>();
        beta = action["beta"].cast<double>();

        kinematics_step(action["dt"].cast<double>());
        update(action["dt"].cast<double>());
        return to_dict();
    }

    virtual void kinematics_step(double dt)
    {
        if (integrator == "euler")
        {
            *this = *this + *this->d(dt) * dt;
        }
        else if (integrator == "midpoint")
        {
            auto temp1 = std::make_shared<Object3D>(*this + *this->d(dt * 0.5));
            *this = *this + *temp1->d(dt) * dt;
        }
        else if (integrator == "rk23")
        {
            auto k1 = this->d(dt);
            auto temp1 = std::make_shared<Object3D>(*this + *k1 * dt * 0.5);
            auto k2 = temp1->d(dt);
            auto temp2 = std::make_shared<Object3D>(*this + *k2 * dt * 0.5);
            auto k3 = temp2->d(dt);
            *this = *this + (*k1 + *k2 * 2 + *k3) * dt / 4;
        }
        else if (integrator == "rk45")
        {
            auto k1 = this->d(dt);
            auto temp1 = std::make_shared<Object3D>(*this + *k1 * dt * 0.5);
            auto k2 = temp1->d(dt);
            auto temp2 = std::make_shared<Object3D>(*this + *k2 * dt * 0.5);
            auto k3 = temp2->d(dt);
            auto temp3 = std::make_shared<Object3D>(*this + *k3 * dt);
            auto k4 = temp3->d(dt);
            *this = *this + (*k1 + *k2 * 2 + *k3 * 2 + *k4) * dt / 6;
        }
    }

    virtual std::shared_ptr<Object3D> d(double dt)
    {
        auto next_object = std::make_shared<Object3D>(*this);
        auto derivative = std::make_shared<Object3D>(*this);

        derivative->pos[0] = vel[0];
        derivative->pos[1] = vel[1];
        derivative->pos[2] = vel[2];

        derivative->vel[0] = 0;
        derivative->vel[1] = 0;
        derivative->vel[2] = 0;

        derivative->ang_vel[0] = 0;
        derivative->ang_vel[1] = 0;
        derivative->ang_vel[2] = 0;

        derivative->theta = ang_vel[1] * sin(gamma) + ang_vel[2] * cos(gamma);
        derivative->phi = (ang_vel[1] * cos(gamma) - ang_vel[2] * sin(gamma)) / cos(theta);
        derivative->gamma = ang_vel[0] * - tan(theta) * (ang_vel[1] * cos(gamma) - ang_vel[2] * sin(gamma));

        derivative->theta_v = 0;
        derivative->phi_v = 0;
        derivative->gamma_v = 0;

        derivative->alpha = 0;
        derivative->beta = 0;

        return derivative;
    }
    
    virtual void update(double dt)
    {
        theta_v = atan2(vel[1], sqrt(vel[0] * vel[0] + vel[2] * vel[2]));
        phi_v = atan2(-vel[2], vel[0]);
        
        beta = cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma);
        alpha = (cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta);
        gamma_v = (cos(alpha) * sin(beta) * sin(theta) - sin(alpha) * sin(beta) * cos(gamma) * cos(theta) + cos(beta) * sin(gamma) * cos(theta)) / cos(theta_v);
    }

    virtual Object3D operator+(const Object3D& other) const
    {
        Object3D result = *this;

        for (int i = 0; i < 3; ++i)
        {   
            result.pos[i] = pos[i] + other.pos[i];
            result.vel[i] = vel[i] + other.vel[i];
            result.ang_vel[i] = ang_vel[i] + other.ang_vel[i];
        }   

        result.theta = theta + other.theta;
        result.phi = phi + other.phi;
        result.gamma = gamma + other.gamma;
        result.theta_v = theta_v + other.theta_v;
        result.phi_v = phi_v + other.phi_v; 
        result.alpha = alpha + other.alpha;
        result.beta = beta + other.beta;
        result.gamma_v = gamma_v + other.gamma_v;


        return result;
    }

    virtual Object3D operator-(const Object3D& other) const
    {
        Object3D result = *this;

        for (int i = 0; i < 3; ++i)
        {
            result.pos[i] = pos[i] - other.pos[i];
            result.vel[i] = vel[i] - other.vel[i];
            result.ang_vel[i] = ang_vel[i] - other.ang_vel[i];
        }

        result.theta = theta - other.theta;
        result.phi = phi - other.phi;
        result.gamma = gamma - other.gamma;
        result.theta_v = theta_v - other.theta_v;
        result.phi_v = phi_v - other.phi_v;
        result.alpha = alpha - other.alpha;
        result.beta = beta - other.beta;
        result.gamma_v = gamma_v - other.gamma_v;

        return result;
    }

    virtual Object3D operator*(const double& other) const
    {
        Object3D result = *this;

        for (int i = 0; i < 3; ++i)
        {
            result.pos[i] = pos[i] * other; 
            result.vel[i] = vel[i] * other;
            result.ang_vel[i] = ang_vel[i] * other;
        }   

        result.theta = theta * other;
        result.phi = phi * other;
        result.gamma = gamma * other;
        result.theta_v = theta_v * other;
        result.phi_v = phi_v * other;
        result.alpha = alpha * other;
        result.beta = beta * other;
        result.gamma_v = gamma_v * other;

        return result;
    }
    
    virtual Object3D operator/(const double& other) const
    {
        Object3D result = *this;

        for (int i = 0; i < 3; ++i)
        {
            result.pos[i] = pos[i] / other; 
            result.vel[i] = vel[i] / other;
            result.ang_vel[i] = ang_vel[i] / other;
        }   

        result.theta = theta / other;
        result.phi = phi / other;
        result.gamma = gamma / other;
        result.theta_v = theta_v / other;
        result.phi_v = phi_v / other;   
        result.alpha = alpha / other;
        result.beta = beta / other;
        result.gamma_v = gamma_v / other;

        return result;
    }
};