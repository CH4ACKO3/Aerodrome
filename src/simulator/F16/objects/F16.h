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
    const double fttom = 0.304878f;
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
    double power0;
    double Ny;
    double Nz;

    double cxt;
    double cyt;
    double czt;
    double clt;
    double cmt;
    double cnt;

    std::string model;

    F16() {}

    F16(py::dict input_dict) : Aircraft3D(input_dict)
    {
        power = input_dict["power"].cast<double>();
        power0 = power;
        model = input_dict["model"].cast<std::string>();
        thtlc = 0.0;
        el = 0.0;
        ail = 0.0;
        rdr = 0.0;
        Ny = 0.0;
        Nz = 0.0;
        cxt = 0.0;
        cyt = 0.0;
        czt = 0.0;
        clt = 0.0;
        cmt = 0.0;
        cnt = 0.0;
    }

    virtual void reset() override
    {
        power = power0;
        thtlc = 0.0;
        el = 0.0;
        ail = 0.0;
        rdr = 0.0;
        Ny = 0.0;
        Nz = 0.0;
        cxt = 0.0;
        cyt = 0.0;
        czt = 0.0;
        clt = 0.0;
        cmt = 0.0;
        cnt = 0.0;
        Aircraft3D::reset();
    }

    virtual py::dict to_dict() override
    {
        py::dict output_dict = Aircraft3D::to_dict();
        output_dict["model"] = model;
        output_dict["power"] = power;
        output_dict["thtlc"] = thtlc;
        output_dict["el"] = el;
        output_dict["ail"] = ail;
        output_dict["rdr"] = rdr;
        output_dict["Ny"] = Ny;
        output_dict["Nz"] = Nz;
        output_dict["ct"] = std::array<double, 6>({cxt, cyt, czt, clt, cmt, cnt});
        return output_dict;
    }

    virtual py::object step(py::dict input_dict) override
    {
        thtlc = input_dict["thtlc"].cast<double>();
        el = input_dict["el"].cast<double>();
        ail = input_dict["ail"].cast<double>();
        rdr = input_dict["rdr"].cast<double>();

        double alpha_ = alpha * F16Val.rtod; // deg
        double beta_ = beta * F16Val.rtod; // deg
        double alt = h; // ft
        double vt = V; // ft/s

        adc_return adc_ret = adc(vt, alt);
        double amach = adc_ret.amach;
        double qbar = adc_ret.qbar; // slug/ft^2

        double cpow = tgear(thtlc);
        power += pdot(power, cpow) * dt;
        T = thrust(power, alt, amach);
        
        double dail = ail / 20.0f;
        double drdr = rdr / 30.0f;

        cxt = cx(alpha_, el);
        cyt = cy(beta_, ail, rdr);
        czt = cz(alpha_, beta_, el);

        clt = cl(alpha_, beta_) + dlda(alpha_, beta_) * dail + dldr(alpha_, beta_) * drdr;
        cmt = cm(alpha_, el);
        cnt = cn(alpha_, beta_) + dnda(alpha_, beta_) * dail + dndr(alpha_, beta_) * drdr;
 
        double tvt = .5f / vt;
        double b2v = F16Val.b * tvt;
        double cq = F16Val.cbar * ang_vel_b(1) * tvt;

        // Damping
        auto d = dampp(alpha_);

        if (model == "stevens")
        {
            cxt = cxt + cq * d[0];
            cyt = cyt + b2v * (d[1] * ang_vel_b(2) + d[2] * ang_vel_b(0));
            czt = czt + cq * d[3];
            clt = clt + b2v * (d[4] * ang_vel_b(2) + d[5] * ang_vel_b(0));
            cmt = cmt + cq * d[6] + czt * (F16Val.xcgr - F16Val.xcg);
            cnt = cnt + b2v * (d[7] * ang_vel_b(2) + d[8] * ang_vel_b(0)) - cyt * (F16Val.xcgr - F16Val.xcg) * F16Val.cbar / F16Val.b;
        }
        else if (model == "morelli")
        {
            Eigen::Matrix<double, 6, 1> result = morelli(alpha, beta, el/F16Val.rtod, ail/F16Val.rtod, rdr/F16Val.rtod, ang_vel_b(0), ang_vel_b(1), ang_vel_b(2), F16Val.cbar, F16Val.b, vt, F16Val.xcg, F16Val.xcgr);
            cxt = result(0);
            cyt = result(1);
            czt = result(2);
            clt = result(3);
            cmt = result(4);
            cnt = result(5);
        }

        double qs = qbar * F16Val.s;
        double qsb = qs * F16Val.b;
        double rmqs = F16Val.rm * qs;
        double ay = rmqs * cyt;
        double az = rmqs * czt;

        D = -qs * cxt; // defined in body frame
        L = -qs * czt;
        N = qs * cyt;

        M[0] = qsb * clt;
        M[1] = qsb * cmt;
        M[2] = qsb * cnt;

        force_vec c_force = {T - D - m * F16Val.g * sin(theta),
                             N + m * F16Val.g * cos(theta) * sin(gamma),
                             -L + m * F16Val.g * cos(theta) * cos(gamma),
                             M[0], M[1], M[2]}; // 力和力矩

        // force_vec c_force = {0, 0, 0, 0, 0, 0};
        kinematics_step(c_force);

        h = -pos(2);

        Tem = Temperature(h * F16Val.fttom);
        Pres = Pressure(h * F16Val.fttom);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(h * F16Val.fttom);
        
        q = 0.5 * Rho * V * V;

        // ground frame
        // Nz = (L * cos(gamma) * cos(theta) - N * sin(gamma) * cos(theta) + (T - D) * sin(theta)) / (m * F16Val.g) - 1;

        // wind frame
        // Ny = (L * sin(gamma) * cos(beta) + N * cos(gamma) * cos(beta) - (T - D) * cos(alpha) * sin(beta)) / (m * F16Val.g);

        // code from stanleybak
        double xa = 15.0;
        az = az-xa*((F16Val.c5 * ang_vel_b(0)-F16Val.c7 * F16Val.he) * ang_vel_b(2) + F16Val.c6 * (ang_vel_b(2) * ang_vel_b(2)-ang_vel_b(0) * ang_vel_b(0)) + qs * F16Val.cbar * F16Val.c7 * cmt);
        // ay = ay+xa*((F16Val.c8 * ang_vel_b(0)-F16Val.c2 * ang_vel_b(2) + F16Val.c9 * F16Val.he) * ang_vel_b(1) + qsb * (F16Val.c4 * clt + F16Val.c9 * cnt));

        Nz = (-az / F16Val.g) - 1;
        Ny = ay / F16Val.g;

        return to_dict();
    }
};

