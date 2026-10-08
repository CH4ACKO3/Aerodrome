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
#include <stdexcept>

using namespace yAtmosphere;
namespace py = pybind11;

struct F16PlantParameters {
    const double xcg = 0.35f;
    const double s = 300.0f;
    const double b = 30.0f;
    const double cbar = 11.32f;
    const double xcgr = .35f;
    const double he = 160.0f;
    const double rtod = 57.29578f;
    const double fttom = 0.3048;
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
        S = F16Val.s;
        c = F16Val.cbar;
        update_atmosphere();
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
        update_atmosphere();
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

        double cpow = tgear(thtlc);
        flight_state_vec current;
        current << pos, vel_b, quat.w(), quat.x(), quat.y(), quat.z(), ang_vel_b, m, power;

        flight_state_vec next;
        if (integrator == "euler")
        {
            next = current + dt * flight_derivative(current, cpow);
        }
        else if (integrator == "midpoint")
        {
            flight_state_vec k1 = flight_derivative(current, cpow);
            next = current + dt * flight_derivative(current + 0.5 * dt * k1, cpow);
        }
        else if (integrator == "rk4")
        {
            flight_state_vec k1 = flight_derivative(current, cpow);
            flight_state_vec k2 = flight_derivative(current + 0.5 * dt * k1, cpow);
            flight_state_vec k3 = flight_derivative(current + 0.5 * dt * k2, cpow);
            flight_state_vec k4 = flight_derivative(current + dt * k3, cpow);
            next = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        else
        {
            throw std::invalid_argument("Unknown F16 integrator: " + integrator);
        }

        state_vec next_body = next.head<14>();
        set_kinematic_state(next_body);
        power = next(14);
        h = -pos(2);
        update_atmosphere();

        state_vec settled = {pos(0), pos(1), pos(2), vel_b(0), vel_b(1), vel_b(2),
                             quat.w(), quat.x(), quat.y(), quat.z(),
                             ang_vel_b(0), ang_vel_b(1), ang_vel_b(2), m};
        FlightLoads loads = evaluate_loads(settled, power);
        T = loads.thrust;
        D = -loads.qs * loads.cx;
        L = -loads.qs * loads.cz;
        N = loads.qs * loads.cy;
        M[0] = loads.force(3);
        M[1] = loads.force(4);
        M[2] = loads.force(5);
        cxt = loads.cx;
        cyt = loads.cy;
        czt = loads.cz;
        clt = loads.cl;
        cmt = loads.cm;
        cnt = loads.cn;

        Eigen::Vector3d omega = settled.segment<3>(10);
        Eigen::Vector3d moment = loads.force.tail<3>();
        Eigen::Vector3d angular_acceleration = J_inv * (moment - omega.cross(J * omega));
        double xa = 15.0; // Pilot station ahead of the center of gravity, in feet.
        Nz = -(loads.qs * loads.cz / m - xa * angular_acceleration(1)) / F16Val.g - 1;
        Ny = (loads.qs * loads.cy / m + xa * angular_acceleration(2)) / F16Val.g;

        return to_dict();
    }

private:
    using flight_state_vec = Eigen::Matrix<double, 15, 1>; // Object3D state followed by engine power.

    struct FlightLoads
    {
        force_vec force;
        double thrust;
        double qs;
        double cx, cy, cz, cl, cm, cn;
    };

    FlightLoads evaluate_loads(const state_vec& state, double engine_power) const
    {
        Eigen::Vector3d velocity = state.segment<3>(3);
        Eigen::Vector3d omega = state.segment<3>(10);
        double vt = velocity.norm(); // ft/s
        double alpha_r = atan2(velocity(2), velocity(0));
        double beta_r = asin(velocity(1) / vt);
        double alpha_d = alpha_r * F16Val.rtod;
        double beta_d = beta_r * F16Val.rtod;
        double alt = -state(2); // ft
        adc_return atmosphere = adc(vt, alt);
        double thrust_value = thrust(engine_power, alt, atmosphere.amach);

        double dail = ail / 20.0f;
        double drdr = rdr / 30.0f;
        double cx_value = cx(alpha_d, el);
        double cy_value = cy(beta_d, ail, rdr);
        double cz_value = cz(alpha_d, beta_d, el);
        double cl_value = cl(alpha_d, beta_d) + dlda(alpha_d, beta_d) * dail + dldr(alpha_d, beta_d) * drdr;
        double cm_value = cm(alpha_d, el);
        double cn_value = cn(alpha_d, beta_d) + dnda(alpha_d, beta_d) * dail + dndr(alpha_d, beta_d) * drdr;

        double tvt = .5f / vt;
        double b2v = F16Val.b * tvt;
        double cq = F16Val.cbar * omega(1) * tvt;

        if (model == "morelli")
        {
            Eigen::Matrix<double, 6, 1> result = morelli(alpha_r, beta_r, el/F16Val.rtod, ail/F16Val.rtod, rdr/F16Val.rtod, omega(0), omega(1), omega(2), F16Val.cbar, F16Val.b, vt, F16Val.xcg, F16Val.xcgr);
            cx_value = result(0);
            cy_value = result(1);
            cz_value = result(2);
            cl_value = result(3);
            cm_value = result(4);
            cn_value = result(5);
        }

        auto damping = dampp(alpha_d);
        cx_value += cq * damping[0];
        cy_value += b2v * (damping[1] * omega(2) + damping[2] * omega(0));
        cz_value += cq * damping[3];
        cl_value += b2v * (damping[4] * omega(2) + damping[5] * omega(0));
        cm_value += cq * damping[6] + cz_value * (F16Val.xcgr - F16Val.xcg);
        cn_value += b2v * (damping[7] * omega(2) + damping[8] * omega(0)) - cy_value * (F16Val.xcgr - F16Val.xcg) * F16Val.cbar / F16Val.b;

        double qs = atmosphere.qbar * F16Val.s;
        double mass = state(13);
        Eigen::Quaterniond attitude(state(6), state(7), state(8), state(9));
        attitude.normalize();
        Eigen::Vector3d gravity_body = attitude.conjugate() * Eigen::Vector3d(0, 0, mass * F16Val.g);

        force_vec force;
        force << thrust_value + qs * cx_value + gravity_body(0),
                 qs * cy_value + gravity_body(1),
                 qs * cz_value + gravity_body(2),
                 qs * F16Val.b * cl_value,
                 qs * F16Val.cbar * cm_value - omega(2) * F16Val.he,
                 qs * F16Val.b * cn_value + omega(1) * F16Val.he;
        return {force, thrust_value, qs, cx_value, cy_value, cz_value, cl_value, cm_value, cn_value};
    }

    flight_state_vec flight_derivative(const flight_state_vec& current, double commanded_power) const
    {
        state_vec body = current.head<14>();
        body.segment<4>(6).normalize();
        FlightLoads loads = evaluate_loads(body, current(14));
        flight_state_vec derivative;
        derivative.head<14>() = d(body, loads.force);
        derivative(14) = pdot(current(14), commanded_power);
        return derivative;
    }

    void update_atmosphere()
    {
        double height_m = h * F16Val.fttom;
        Tem = Temperature(height_m);
        Pres = Pressure(height_m);
        Rho = Density(Tem, Pres);
        a = SpeedofSound(Tem);
        g = Gravity(height_m);
        double speed_m_s = V * F16Val.fttom;
        q = 0.5 * Rho * speed_m_s * speed_m_s;
    }
};
