#pragma once

#include "BaseEnv.h"
#include "Object3D.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;

class Space3D : public BaseEnv
{
protected:
    std::vector<std::shared_ptr<Object3D>> objects;
    double tau;
    double dt;
    double eps;
    std::string integrator;
    int integrate_steps;

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

    Space3D(double tau, double eps, std::string integrator, int integrate_steps)
        : tau(tau), eps(eps), integrator(integrator), integrate_steps(integrate_steps)
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

    py::object reset()
    {   
        return py::none();
    }

    py::object step(const py::object& action)
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