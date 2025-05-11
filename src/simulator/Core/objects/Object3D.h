#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <array>
#include <cmath>

namespace py = pybind11;

class Object3D {
public:
    std::string name;
    std::string integrator; // Integrator type
    double dt;

    double init_m, m, d_m; // Initial mass; mass; mass change rate
    Eigen::Matrix3d J; // Inertia matrix
    Eigen::Matrix3d J_inv; // Inverse of inertia matrix

    Eigen::Vector3d init_pos, pos; // Position (ground frame)
    Eigen::Vector3d init_vel_g, vel_g; // Velocity (ground frame)
    Eigen::Vector3d init_vel_b, vel_b; // Velocity (body frame)
    Eigen::Vector3d init_ang_vel_b, ang_vel_b; // Angular velocity (body frame)
    Eigen::Vector3d init_ang_g, ang_g; // Euler angles (ground frame, roll, pitch, yaw)
    Eigen::Quaterniond quat; // Quaternion (ground frame)

    double init_V, V; // Velocity
    double init_theta, theta;    // Pitch angle, north-up-east
    double init_phi, phi;      // Yaw angle
    double init_gamma, gamma;    // Roll angle
    double init_theta_v, theta_v;  // Velocity pitch angle
    double init_phi_v, phi_v;    // Velocity yaw angle
    double alpha;    // Angle of attack
    double beta;     // Sideslip angle
    // double gamma_v;  // Velocity roll angle

    typedef Eigen::Matrix<double, 14, 1> state_vec; // p_n, p_e, p_d, v_b_i, v_b_j, v_b_k, e_0, e_1, e_2, e_3, omega_b_i, omega_b_j, omega_b_k, m
    typedef Eigen::Matrix<double, 6, 1> force_vec; // fx, fy, fz, mx, my, mz

    Object3D() {}

    Object3D(py::dict input_dict)
    {
        name = input_dict["name"].cast<std::string>();
        integrator = input_dict["integrator"].cast<std::string>();
        dt = input_dict["dt"].cast<double>();

        init_m = m = input_dict["m"].cast<double>();
        d_m = 0;

        init_pos = pos = input_dict["pos"].cast<Eigen::Vector3d>();

        std::array<double, 3> vel_g_ = input_dict["vel"].cast<std::array<double, 3>>();
        init_vel_g = vel_g = Eigen::Map<Eigen::Vector3d>(vel_g_.data());

        std::array<double, 3> ang_vel_b_ = input_dict["ang_vel"].cast<std::array<double, 3>>();
        init_ang_vel_b = ang_vel_b = Eigen::Map<Eigen::Vector3d>(ang_vel_b_.data());

        std::array<double, 9> J_ = input_dict["J"].cast<std::array<double, 9>>();
        J = Eigen::Map<Eigen::Matrix3d>(J_.data());
        J_inv = J.inverse();

        init_V = V = sqrt(vel_g(0) * vel_g(0) + vel_g(1) * vel_g(1) + vel_g(2) * vel_g(2));
        init_theta = theta = input_dict["theta"].cast<double>(); // north-up-east
        init_phi = phi = input_dict["phi"].cast<double>();
        init_gamma = gamma = input_dict["gamma"].cast<double>();
        init_theta_v = theta_v = input_dict["theta_v"].cast<double>();
        init_phi_v = phi_v = input_dict["phi_v"].cast<double>();

        init_ang_g = ang_g = Eigen::Vector3d(init_gamma, init_theta, -init_phi); // ned

        beta = asin(cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma));
        alpha = asin((cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta));

        init_vel_b = vel_b = Eigen::Vector3d(V * cos(alpha) * cos(beta), V * sin(beta), V * sin(alpha) * cos(beta));

        quat = Eigen::Quaterniond(Eigen::AngleAxisd(init_ang_g(2), Eigen::Vector3d::UnitZ()) *
                                   Eigen::AngleAxisd(init_ang_g(1), Eigen::Vector3d::UnitY()) *
                                   Eigen::AngleAxisd(init_ang_g(0), Eigen::Vector3d::UnitX()));
    }

    virtual void reset()
    {
        pos = init_pos;
        vel_g = init_vel_g;
        ang_vel_b = init_ang_vel_b;
        m = init_m;
        d_m = 0;

        V = init_V;
        theta = init_theta;
        phi = init_phi;
        gamma = init_gamma;
        theta_v = init_theta_v;
        phi_v = init_phi_v;

        ang_g = init_ang_g;

        beta = asin(cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma));
        alpha = asin((cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta));

        vel_b = init_vel_b;
        quat = Eigen::Quaterniond(Eigen::AngleAxisd(init_ang_g(2), Eigen::Vector3d::UnitZ()) *
                                   Eigen::AngleAxisd(init_ang_g(1), Eigen::Vector3d::UnitY()) *
                                   Eigen::AngleAxisd(init_ang_g(0), Eigen::Vector3d::UnitX()));
    }

    virtual py::dict to_dict()
    {
        py::dict output_dict;
        output_dict["pos"] = pos;
        output_dict["vel_g"] = vel_g;
        output_dict["vel_b"] = vel_b;
        output_dict["ang_vel_b"] = ang_vel_b;
        output_dict["ang_g"] = ang_g;   
        output_dict["quat"] = quat.coeffs();
        output_dict["m"] = m;
        output_dict["dt"] = dt;
        output_dict["J"] = J;
        output_dict["J_inv"] = J_inv;
        output_dict["V"] = V;
        output_dict["theta"] = theta;
        output_dict["phi"] = phi;
        output_dict["gamma"] = gamma;
        output_dict["theta_v"] = theta_v;
        output_dict["phi_v"] = phi_v;
        output_dict["alpha"] = alpha;
        output_dict["beta"] = beta;
        output_dict["name"] = name;
        return output_dict;
    }

    virtual py::object step(py::dict action)
    {
        double fx = 0, fy = 0, fz = 0; // 力 (弹体系)
        double mx = 0, my = 0, mz = 0; // 力矩 (弹体系)

        force_vec c_force = {fx, fy, fz, mx, my, mz}; // 力和力矩
        kinematics_step(c_force); // 更新状态

        return to_dict();
    }
    
    void kinematics_step(const force_vec& c_force)
    {
        state_vec c_state = {pos(0), pos(1), pos(2), vel_b(0), vel_b(1), vel_b(2), quat.w(), quat.x(), quat.y(), quat.z(), ang_vel_b(0), ang_vel_b(1), ang_vel_b(2), m};
        state_vec new_state;
        new_state.setZero();

        if (integrator == "euler")
        {
            // Update state using Euler method
            state_vec d_state = d(c_state, c_force);

            new_state = c_state + d_state * dt;
        }
        else if (integrator == "midpoint")
        {
            // Midpoint method
            state_vec k1 = d(c_state, c_force);
            state_vec k2 = d(c_state + k1 * (dt / 2), c_force);

            new_state = c_state + k2 * dt;
        }
        else if (integrator == "rk4")
        {
            // RK4 method
            state_vec k1 = d(c_state, c_force);
            state_vec k2 = d(c_state + k1 * (dt / 2), c_force);
            state_vec k3 = d(c_state + k2 * (dt / 2), c_force);
            state_vec k4 = d(c_state + k3 * dt, c_force);

            new_state = c_state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6);
        }

        pos(0) = new_state(0);
        pos(1) = new_state(1);
        pos(2) = new_state(2);
        vel_b(0) = new_state(3);
        vel_b(1) = new_state(4);
        vel_b(2) = new_state(5);
        quat = Eigen::Quaterniond(new_state(6), new_state(7), new_state(8), new_state(9));
        ang_vel_b(0) = new_state(10);
        ang_vel_b(1) = new_state(11);
        ang_vel_b(2) = new_state(12);
        m = new_state(13);

        ang_g = Eigen::Vector3d(atan2(2*(quat.w()*quat.x() + quat.y()*quat.z()), quat.w()*quat.w() + quat.z()*quat.z() - quat.x()*quat.x() - quat.y()*quat.y()),
                                asin(2*(quat.w()*quat.y() - quat.x()*quat.z())),
                                atan2(2*(quat.w()*quat.z() + quat.x()*quat.y()), quat.w()*quat.w() + quat.x()*quat.x() - quat.y()*quat.y() - quat.z()*quat.z()));

        V = sqrt(vel_b(0) * vel_b(0) + vel_b(1) * vel_b(1) + vel_b(2) * vel_b(2));

        vel_g = quat.toRotationMatrix() * vel_b;

        theta = ang_g(1);
        phi = -ang_g(2);
        gamma = ang_g(0);

        theta_v = asin(-vel_g(2) / V);
        phi_v = atan2(-vel_g(1), vel_g(0));

        beta = asin(cos(theta_v) * (cos(gamma) * sin(phi - phi_v) + sin(theta) * sin(gamma) * cos(phi - phi_v)) - sin(theta_v) * cos(theta) * sin(gamma));
        alpha = asin((cos(theta_v) * (sin(theta) * cos(gamma) * cos(phi - phi_v) - sin(gamma) * sin(phi - phi_v)) - sin(theta_v) * cos(theta) * cos(gamma)) / cos(beta));
    }

    state_vec d(const state_vec& c_state, const force_vec& c_force) const
    {
        // force and moment are defined in body frame

        // c_x: current value of x
        // d_x: derivative of x

        // auto c_p_n = c_state(0);
        // auto c_q_e = c_state(1);
        // auto c_r_d = c_state(2);
        // auto c_u = c_state(3);
        // auto c_v = c_state(4);
        // auto c_w = c_state(5);
        // auto c_e_0 = c_state(6);
        // auto c_e_1 = c_state(7);
        // auto c_e_2 = c_state(8);
        // auto c_e_3 = c_state(9);
        // auto c_p = c_state(10);
        // auto c_q = c_state(11);
        // auto c_r = c_state(12);
        // auto c_m = c_state(13);

        auto [c_p_n, c_p_e, c_p_d, c_u, c_v, c_w, c_e_0, c_e_1, c_e_2, c_e_3, c_p, c_q, c_r, c_m] = 
            std::array{c_state(0), c_state(1), c_state(2), c_state(3), c_state(4), c_state(5), 
                      c_state(6), c_state(7), c_state(8), c_state(9), c_state(10), c_state(11), 
                      c_state(12), c_state(13)};

        // double c_fx = c_force(0), c_fy = c_force(1), c_fz = c_force(2);
        // double c_mx = c_force(3), c_my = c_force(4), c_mz = c_force(5);

        auto [c_fx, c_fy, c_fz, c_mx, c_my, c_mz] = 
            std::array{c_force(0), c_force(1), c_force(2), c_force(3), c_force(4), c_force(5)};


        auto d_p_n = (c_e_1 * c_e_1 + c_e_0 * c_e_0 - c_e_2 * c_e_2 - c_e_3 * c_e_3) * c_u + 2 * (c_e_1 * c_e_2 - c_e_3 * c_e_0) * c_v         + 2 * (c_e_1 * c_e_3 + c_e_0 * c_e_2) * c_w;
        auto d_p_e = 2 * (c_e_1 * c_e_2 + c_e_3 * c_e_0) * c_u         + (c_e_2 * c_e_2 + c_e_0 * c_e_0 - c_e_1 * c_e_1 - c_e_3 * c_e_3) * c_v + 2 * (c_e_2 * c_e_3 - c_e_1 * c_e_0) * c_w;
        auto d_p_d = 2 * (c_e_1 * c_e_3 - c_e_2 * c_e_0) * c_u         + 2 * (c_e_2 * c_e_3 + c_e_1 * c_e_0) * c_v         + (c_e_3 * c_e_3 + c_e_0 * c_e_0 - c_e_1 * c_e_1 - c_e_2 * c_e_2) * c_w;

        auto d_u = c_r * c_v - c_q * c_w + c_fx / c_m;
        auto d_v = c_p * c_w - c_r * c_u + c_fy / c_m;
        auto d_w = c_q * c_u - c_p * c_v + c_fz / c_m;

        Eigen::Vector3d c_ang_vel(c_p, c_q, c_r);
        Eigen::Vector3d c_moment(c_mx, c_my, c_mz);
        Eigen::Vector3d d_ang_vel = J_inv * (c_moment - c_ang_vel.cross(J * c_ang_vel));

        double lda = 1000.0, square_norm_e = c_e_0 * c_e_0 + c_e_1 * c_e_1 + c_e_2 * c_e_2 + c_e_3 * c_e_3;
        double tmp = lda * (1 - square_norm_e);

        auto d_e_0 = 0.5 * (c_e_0 * tmp - c_e_1 * c_p - c_e_2 * c_q - c_e_3 * c_r);
        auto d_e_1 = 0.5 * (c_e_0 * c_p + c_e_1 * tmp + c_e_2 * c_r - c_e_3 * c_q);
        auto d_e_2 = 0.5 * (c_e_0 * c_q - c_e_1 * c_r + c_e_2 * tmp + c_e_3 * c_p);
        auto d_e_3 = 0.5 * (c_e_0 * c_r + c_e_1 * c_q - c_e_2 * c_p + c_e_3 * tmp);

        state_vec d_state = {d_p_n, d_p_e, d_p_d,
                             d_u, d_v, d_w, 
                             d_e_0, d_e_1, d_e_2, d_e_3,
                             d_ang_vel(0), d_ang_vel(1), d_ang_vel(2), d_m};
        
        return d_state;
    }
};