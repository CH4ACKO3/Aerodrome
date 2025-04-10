#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Aircraft3D.h"
#include "y_atmosphere.h"
#include "LowLevelFunctions.h"
#include <vector>
#include <string>
#include <array>
#include <cmath>

using namespace yAtmosphere;
namespace py = pybind11;

struct F16PlantParameters {
    const double xcg = 0.35f;
    const double s = 300.0f;
    const double b = 30.0f;
    const double cbar = 11.32f;
    const double rm = 1.57e-3f;
    const double xcgr = .35f;
    const double he = 160.0f;
    const double c1 = -.770f;
    const double c2 = .02755f;
    const double c3 = 1.055e-4f;
    const double c4 = 1.642e-6f;
    const double c5 = .9604f;
    const double c6 = 1.759e-2f;
    const double c7 = 1.792e-5f;
    const double c8 = -.7336f;
    const double c9 = 1.587e-5f;
    const double rtod = 57.29578f;
    const double mtoft = 3.28f;
    const double fttom = 0.3048f;
    const double g = 32.17f;
};

F16PlantParameters F16Val = F16PlantParameters();

class F16 : public Aircraft3D
{
public:
    double thtlc; // throttle lever position
    double el;    // elevator deflection
    double ail;   // aileron deflection
    double rdr;   // rudder deflection

    double power; // engine power
    double Ny;

    F16() {}

    F16(py::dict input_dict) : Aircraft3D(input_dict)
    {
        power = 0.0;
        Ny = 0.0;
    }

    virtual void reset() override
    {
        power = 0.0;
        Ny = 0.0;
        Aircraft3D::reset();
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = Aircraft3D::to_dict();
        output_dict["power"] = power;
        output_dict["Ny"] = Ny;
        return output_dict;
    }

    virtual py::object step(py::dict input_dict) override
    {
        double thtlc = input_dict["thtlc"].cast<double>();
        double el = input_dict["el"].cast<double>();
        double ail = input_dict["ail"].cast<double>();
        double rdr = input_dict["rdr"].cast<double>();

        double alpha_ = alpha * F16Val.rtod;
        double beta_ = beta * F16Val.rtod;
        double alt = h;
        double vt = V;

        adc_return adc_ret = adc(vt, alt);
        double amach = adc_ret.amach;
        double qbar = adc_ret.qbar;

        double cpow = tgear(thtlc);
        power += pdot(power, cpow) * dt;
        T = thrust(power, alt, amach);
        
        double dail = ail / 20.0f;
        double drdr = rdr / 30.0f;

        double cxt = cx(alpha_, el);
        double cyt = cy(beta_, ail, rdr);
        double czt = cz(alpha_, beta_, el);

        double clt = cl(alpha_, beta_) + dlda(alpha_, beta_) * dail + dldr(alpha_, beta_) * drdr;
        double cmt = cm(alpha_, el);
        double cnt = cn(alpha_, beta_) + dnda(alpha_, beta_) * dail + dndr(alpha_, beta_) * drdr;
 
        double tvt = .5f / vt;
        double b2v = F16Val.b * tvt;
        double cq = F16Val.cbar * ang_vel(2) * tvt;

        // Damping
        auto d = dampp(alpha_);
        cxt = cxt + cq * d[0];
        cyt = cyt + b2v * (d[1] * -ang_vel(1) + d[2] * ang_vel(0));
        czt = czt + cq * d[3];
        clt = clt + b2v * (d[4] * -ang_vel(1) + d[5] * ang_vel(0));
        cmt = cmt + cq * d[6] + czt * (F16Val.xcgr - F16Val.xcg);
        cnt = cnt + b2v * (d[7] * -ang_vel(1) + d[8] * ang_vel(0)) - cyt * (F16Val.xcgr - F16Val.xcg) * F16Val.cbar / F16Val.b;

        double qs = qbar * F16Val.s;
        double qsb = qs * F16Val.b;
        double rmqs = F16Val.rm * qs;
        double ay = rmqs * cyt;
        double az = rmqs * czt;

        D = qs * cxt;
        L = -qs * czt;
        N = qs * cyt;

        M[0] = qsb * clt;
        M[1] = -qsb * cnt;
        M[2] = qsb * cmt;

        force_vec c_force = {T * cos(alpha) * cos(beta) - D - m * F16Val.g * sin(theta),
            T * (sin(alpha) * cos(gamma_v) + cos(alpha) * sin(beta) * sin(gamma_v)) + L * cos(gamma_v) - N * sin(gamma_v) - m * F16Val.g * cos(theta),
            T * (sin(alpha) * sin(gamma_v) - cos(alpha) * sin(beta) * cos(gamma_v)) + L * sin(gamma_v) + N * cos(gamma_v),
            M[0], M[1], M[2]}; // 力和力矩

        // force_vec c_force = {0, 0, 0, 0, 0, 0};
        kinematics_step(c_force);

        h = pos[1];

        Tem = Temperature(h);
        Pres = Pressure(h);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h);
        
        q = 0.5 * Rho * V * V;

        Ny = (T * (sin(alpha) * cos(gamma_v) - cos(alpha) * sin(beta) * sin(gamma_v))
                                + L * cos(gamma_v) - N * sin(gamma_v) - m * F16Val.g * cos(theta_v)) / (m * F16Val.g);

        return to_dict();
    }
};

