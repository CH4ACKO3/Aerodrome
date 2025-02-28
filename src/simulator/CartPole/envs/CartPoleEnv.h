#pragma once

#include "BaseEnv.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <cmath>
#include <random>

namespace py = pybind11;

class CartPoleEnv : public BaseEnv
{
private:
    double gravity = 9.81;
    double masscart = 1.0;
    double masspole = 0.1;
    double total_mass = masspole + masscart;
    double length = 0.5; // half the pole's length
    double polemass_length = masspole * length;
    double force_mag = 10.0;
    double tau = 0.02; // seconds between state updates
    std::string kinematic_integrator = "euler";

    double theta_threshold_radians = 12 * 2 * 3.1415926 / 360; // angle at which to fail the episode
    double x_threshold = 2.4; // distance at which to fail the episode

    double x = 0;
    double theta = 0;
    double x_dot = 0;
    double theta_dot = 0;

    int time_step = 0;
    int max_steps = 200;
    bool steps_beyond_done = false;

public:
    CartPoleEnv() {}

    ~CartPoleEnv() {}

    py::object reset() override
    {
        py::dict result;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.05, 0.05);

        x = dis(gen);
        x_dot = dis(gen);
        theta = dis(gen);
        theta_dot = dis(gen);
        time_step = 0;
        steps_beyond_done = false;

        result["observation"] = py::make_tuple(x, x_dot, theta, theta_dot);
        result["info"] = "";
        return result;
    }

    py::object step(const py::object& input_dict) override
    {
        py::dict result;
        if (!input_dict.contains("action"))
        {
            result["info"] = "input_dict does not contain 'action'";
            return result;
        }

        int action;
        try
        {
            action = input_dict["action"].cast<int>();
        }
        catch (const std::exception &e)
        {
            result["info"] = std::string("failed to convert action to int: ") + e.what();
            return result;
        }

        if (action != 0 && action != 1)
        {
            result["info"] = "action must be either 0 or 1";
            return result;
        }

        double force = action * force_mag;
        double costheta = cos(theta);
        double sintheta = sin(theta);

        // For the interested reader:
        // https://coneural.org/florian/papers/05_cart_pole.pdf
        double temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass;
        double theta_acc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass));
        double x_acc = temp - polemass_length * theta_acc * costheta / total_mass;

        if (kinematic_integrator == "euler")
        {
            x = x + tau * x_dot;
            x_dot = x_dot + tau * x_acc;
            theta = theta + tau * theta_dot;
            theta_dot = theta_dot + tau * theta_acc;
        }
        else if (kinematic_integrator == "semi-implicit-euler")
        {
            x_dot = x_dot + tau * x_acc;
            x = x + tau * x_dot;
            theta_dot = theta_dot + tau * theta_acc;
            theta = theta + tau * theta_dot;
        }
        else
        {
            result["info"] = "unknown kinematic integrator";
            return result;
        }

        time_step ++;
        bool terminated = ((x < -x_threshold) || (x > x_threshold) || (theta < -theta_threshold_radians) || (theta > theta_threshold_radians));
        bool truncated = (time_step >= max_steps);

        double reward;
        if (!terminated && !truncated)
        {
            reward = 1.0;
        }
        else if (!steps_beyond_done)
        {
            if (terminated)
            {
                reward = 0.0;
            }
            else if (truncated)
            {
                reward = 1.0;
            }
            steps_beyond_done = true;
        }
        else
        {
            reward = 0.0;
            result["info"] = "You are calling 'step()' even though this environment has already returned terminated = True. "
                              "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.";
        }

        result["observation"] = py::make_tuple(x, x_dot, theta, theta_dot);
        result["reward"] = reward;
        result["terminated"] = terminated;
        result["truncated"] = truncated;

        if (!result.contains("info"))
        {
            result["info"] = "";
        }

        return result;
    }
};