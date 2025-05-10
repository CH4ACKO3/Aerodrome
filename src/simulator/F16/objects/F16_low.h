#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "F16.h"
#include "y_atmosphere.h"
#include "LowLevelFunctions.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

class F16_low : public F16
{
public:
    double ThrottleMax;
    double ThrottleMin;
    double ElevatorMaxDeg;
    double ElevatorMinDeg;
    double AileronMaxDeg;
    double AileronMinDeg;
    double RudderMaxDeg;
    double RudderMinDeg;
    double NzMax;
    double NzMin;

    Eigen::Matrix<double, 1, 3> k_long;
    Eigen::Matrix<double, 2, 5> k_lat;
    Eigen::Matrix<double, 13, 1> xequil;
    Eigen::Matrix<double, 4, 1> uequil;
    Eigen::Matrix<double, 3, 8> K_lqr;
    Eigen::Matrix<double, 4, 1> u_deg;

    // Integral states
    double int_e_Ny;
    double int_e_ps;
    double int_e_Nz;

    F16_low() {}

    F16_low(py::dict input_dict) : F16(input_dict)
    {
        ThrottleMax = 1; // Afterburner on for throttle > 0.7
        ThrottleMin = 0;
        ElevatorMaxDeg = 25;
        ElevatorMinDeg = -25;
        AileronMaxDeg = 21.5;
        AileronMinDeg = -21.5;
        RudderMaxDeg = 30;
        RudderMinDeg = -30;
        NzMax = 6;
        NzMin = -1;

        k_long = (Eigen::Matrix<double, 1, 3>() << -156.8801506723475, -31.037008068526642, -38.72983346216317).finished();
    
        k_lat = (Eigen::Matrix<double, 2, 5>() << 
            37.84483, -25.40956, -6.82876, -332.88343, -17.15997,
            -23.91233, 5.69968, -21.63431, 64.49490, -88.36203).finished();

        xequil = (Eigen::Matrix<double, 13, 1>() << 502.0, 0.0389, 0.0, 0.0, 0.0389, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.0567).finished();

        uequil = (Eigen::Matrix<double, 4, 1>() << 0.1395, -0.7496, 0.0, 0.0).finished();

        K_lqr = Eigen::Matrix<double, 3, 8>::Zero();
        K_lqr.block<1, 3>(0, 0) = k_long;
        K_lqr.block<2, 5>(1, 3) = k_lat;

        u_deg = Eigen::Matrix<double, 4, 1>::Zero();

        int_e_Ny = 0;
        int_e_ps = 0;
        int_e_Nz = 0;
    }

    virtual void reset() override
    {
        F16::reset();
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = F16::to_dict();
        output_dict["u_deg"] = u_deg;
        return output_dict;
    }

    virtual py::object step(py::dict input_dict) override
    {
        double Nyc = input_dict["Ny"].cast<double>();
        double ps = input_dict["ps"].cast<double>();
        double Nzc = input_dict["Nz"].cast<double>();
        double throttle = input_dict["throttle"].cast<double>();

        Eigen::Matrix<double, 8, 1> x_ctrl = (Eigen::Matrix<double, 8, 1>() << 
            alpha - xequil(1),
            ang_vel(2) - xequil(7),
            int_e_Ny,
            beta - xequil(2),
            ang_vel(0) - xequil(6),
            -ang_vel(1) - xequil(8),
            int_e_ps,
            int_e_Nz
        ).finished();

        u_deg.setZero();
        u_deg(0) = throttle;
        u_deg.block<3, 1>(1, 0) = -K_lqr * x_ctrl;

        u_deg += uequil;

        u_deg(0) = std::clamp(u_deg(0), ThrottleMin, ThrottleMax);    
        u_deg(1) = std::clamp(u_deg(1), ElevatorMinDeg, ElevatorMaxDeg);
        u_deg(2) = std::clamp(u_deg(2), AileronMinDeg, AileronMaxDeg);
        u_deg(3) = std::clamp(u_deg(3), RudderMinDeg, RudderMaxDeg);

        double thtlc = u_deg(0);
        double el = u_deg(1);
        double ail = u_deg(2);
        double rdr = u_deg(3);

        input_dict["thtlc"] = thtlc;
        input_dict["el"] = el;
        input_dict["ail"] = ail;
        input_dict["rdr"] = rdr;

        F16::step(input_dict);  

        int_e_Ny += dt * (Ny - Nyc);
        int_e_ps += dt * (gamma_v - ps);
        int_e_Nz += dt * (Nz - Nzc);

        return to_dict();
    }
};

