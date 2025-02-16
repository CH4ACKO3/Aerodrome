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

    virtual py::object step(py::dict input_dict)
    {
        return py::none();
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

    virtual Object3D operator+(const Object3D& other) const
    {
        Object3D result;
        for (int i = 0; i < 3; ++i)
        {   
            result.pos[i] = pos[i] + other.pos[i];
            result.vel[i] = vel[i] + other.vel[i];
            result.acc[i] = acc[i] + other.acc[i];
            result.ang_vel[i] = ang_vel[i] + other.ang_vel[i];
            result.ang_acc[i] = ang_acc[i] + other.ang_acc[i];
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
        Object3D result;
        
        for (int i = 0; i < 3; ++i)
        {
            result.pos[i] = pos[i] - other.pos[i];
            result.vel[i] = vel[i] - other.vel[i];
            result.acc[i] = acc[i] - other.acc[i];
            result.ang_vel[i] = ang_vel[i] - other.ang_vel[i];
            result.ang_acc[i] = ang_acc[i] - other.ang_acc[i];
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
        Object3D result;
        for (int i = 0; i < 3; ++i)
        {
            result.pos[i] = pos[i] * other; 
            result.vel[i] = vel[i] * other;
            result.acc[i] = acc[i] * other;
            result.ang_vel[i] = ang_vel[i] * other;
            result.ang_acc[i] = ang_acc[i] * other;
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
        Object3D result;
        for (int i = 0; i < 3; ++i)
        {
            result.pos[i] = pos[i] / other; 
            result.vel[i] = vel[i] / other;
            result.acc[i] = acc[i] / other;
            result.ang_vel[i] = ang_vel[i] / other;
            result.ang_acc[i] = ang_acc[i] / other;
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

