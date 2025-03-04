import numpy as np
from math import pi
from aerodrome.registration import register

TABLE4 = [
    [00000, -0.0065, 288.150, 1.01325000000000E+5],
    [11000, 0.0000, 216.650, 2.26320639734629E+4],
    [20000, 0.0010, 216.650, 5.47488866967777E+3],
    [32000, 0.0028, 228.650, 8.68018684755228E+2],
    [47000, 0.0000, 270.650, 1.10906305554966E+2],
    [51000, -0.0028, 270.650, 6.69388731186873E+1],
    [71000, -0.0020, 214.650, 3.95642042804073E+0],
    [84852, 0.0000, 186.946, 3.73383589976215E-1]
]

def Temperature(z):
    H = z * 6356766 / (z + 6356766)
    for b in range(7):
        if H < TABLE4[b+1][0]:
            break
    return TABLE4[b][2] + TABLE4[b][1] * (H - TABLE4[b][0])

def Pressure(z):
    H = z * 6356766 / (z + 6356766)
    for b in range(7):
        if H < TABLE4[b+1][0]:
            break
    C = -0.0341631947363104
    Hb = TABLE4[b][0]
    Lb = TABLE4[b][1]
    Tb = TABLE4[b][2]
    Pb = TABLE4[b][3]
    if abs(Lb) > 1E-12:
        return Pb * pow(1 + Lb/Tb * (H-Hb), C/Lb)
    else:
        return Pb * np.exp(C * (H-Hb)/Tb)

def Density(T, P):
    return P * 0.00348367635597379 / T

def SpeedofSound(T):
    return np.sqrt(401.87430086589 * T)

def Gravity(z):
    return 9.80665 * pow(1 + z/6356766, -2)

class WingedCone_py:
    def __init__(self, input_dict):
        # Object3D
        self.name = input_dict["name"]
        self.integrator = input_dict["integrator"]
        self.pos = input_dict["pos"]
        self.vel = input_dict["vel"]
        self.ang_vel = input_dict["ang_vel"]
        self.J = input_dict["J"]
        self.V = np.sqrt(self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1] + self.vel[2] * self.vel[2])
        self.theta = input_dict["theta"]
        self.phi = input_dict["phi"]
        self.gamma = input_dict["gamma"]
        self.theta_v = input_dict["theta_v"]
        self.phi_v = input_dict["phi_v"]
        self.gamma_v = input_dict["gamma_v"]
        self.alpha = input_dict["alpha"]
        self.beta = input_dict["beta"]

        # Aircraft3D
        self.V = np.sqrt(self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1] + self.vel[2] * self.vel[2])
        self.h = self.pos[1]
        self.S = input_dict["S"]
        self.c = input_dict["c"]
        self.m = input_dict["m"]

        self.Tem = Temperature(self.h)
        self.Pres = Pressure(self.h)
        self.Rho = Density(self.Tem, self.Pres)
        self.a = SpeedofSound(self.Tem)
        self.g = Gravity(self.h)
        
        self.q = 0.5 * self.Rho * self.V * self.V

        self.L = 0.0
        self.D = 0.0
        self.N = 0.0
        self.T = 0.0
        self.M = [0.0, 0.0, 0.0]

        # WingedCone2D
        self.delta_e = 0.0

        # WingedCone2D_Classic
        self.Kiz = input_dict["Kiz"]
        self.Kwz = input_dict["Kwz"]
        self.Kaz = input_dict["Kaz"]
        self.Kpz = input_dict["Kpz"]

        self.Kp_V = input_dict["Kp_V"]
        self.Ki_V = input_dict["Ki_V"]
        self.Kd_V = input_dict["Kd_V"]

        self.eNy = 0.0
        self.i_eNy = 0.0
        self.p_eNy = 0.0
        self.i_eSAC = 0.0
        self.i_V = 0.0
        self.d_eV = 0.0
        self.eV_prev = 0.0

        self._D()
        self._L()
        self._T()
        self._M()

        self.Ny = (self.T * (np.sin(self.alpha) * np.cos(self.gamma_v) - np.cos(self.alpha) * np.sin(self.beta) * np.sin(self.gamma_v))
                                + self.L * np.cos(self.gamma_v) - self.N * np.sin(self.gamma_v) - self.m * self.g * np.cos(self.theta_v)) / (self.m * self.g)
        self.wz = self.ang_vel[2]

        self.initial_state = input_dict

    def reset(self):
        self.__init__(self.initial_state)

    def _D(self):
        CD = 0.645 * self.alpha * self.alpha + 0.0043378 * self.alpha + 0.003772
        self.D = self.q * self.S * CD

    def _L(self):
        CL = 0.6203 * self.alpha + 2.4 * np.sin(0.08 * self.alpha)
        self.L = self.q * self.S * CL

    def _T(self):
        self.T = 4.959e3

    def _M(self):
        CM1 = -0.035 * self.alpha * self.alpha + 0.036617 * self.alpha + 5.3261e-6
        CM2 = self.ang_vel[2] * self.c * (-6.796 * self.alpha * self.alpha + 0.3015 * self.alpha - 0.2289) / (2 * self.V)
        CM3 = 0.0292 * (self.delta_e - self.alpha)
        self.M[2] = self.q * self.S * self.c * (CM1 + CM2 + CM3)
    
    def to_dict(self):
        return {
            # Object3D
            "name": self.name,
            "integrator": self.integrator,
            "pos": self.pos,
            "vel": self.vel,
            "ang_vel": self.ang_vel,
            "J": self.J,
            "V": np.sqrt(self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1] + self.vel[2] * self.vel[2]),
            "theta": self.theta,
            "phi": self.phi,
            "gamma": self.gamma,
            "theta_v": self.theta_v,
            "phi_v": self.phi_v,
            "gamma_v": self.gamma_v,
            "alpha": self.alpha,
            "beta": self.beta,
            "h": self.pos[1],
            "S": self.S,
            "c": self.c,
            "m": self.m,
            "Tem": self.Tem,
            "Pres": self.Pres,
            "Rho": self.Rho,
            "a": self.a,
            "g": self.g,
            "q": self.q,
            "L": self.L,
            "D": self.D,
            "N": self.N,
            "T": self.T,
            "M": self.M,
            "delta_e": self.delta_e,
            "Kiz": self.Kiz,
            "Kwz": self.Kwz,
            "Kaz": self.Kaz,
            "Kpz": self.Kpz,
            "Kp_V": self.Kp_V,
            "Ki_V": self.Ki_V,
            "Kd_V": self.Kd_V,
            "eNy": self.eNy,
            "i_eNy": self.i_eNy,
            "p_eNy": self.p_eNy,
            "i_eSAC": self.i_eSAC,
            "i_V": self.i_V,
            "d_eV": self.d_eV,
            "eV_prev": self.eV_prev,
            "Ny": self.Ny,
            "wz": self.wz
        }

    def d(self, state):
        pos = state[:3]
        vel = state[3:6]
        ang_vel = state[6:9]
        V, theta, phi, gamma, theta_v, phi_v, alpha, beta, gamma_v = state[9:]

        d_V = (self.T * np.cos(alpha) * np.cos(beta) - self.D - self.m * self.g * np.sin(theta_v)) / self.m
        d_theta_v = (self.T * (np.sin(alpha) * np.cos(gamma_v) - np.cos(alpha) * np.sin(beta) * np.sin(gamma_v))
                                + self.L * np.cos(gamma_v) - self.N * np.sin(gamma_v) - self.m * self.g * np.cos(theta_v)) / (self.m * V)
        d_phi_v = -(self.T * (np.sin(alpha) * np.sin(gamma_v) - np.cos(alpha) * np.sin(beta) * np.cos(gamma_v))
                            + self.L * np.sin(gamma_v) + self.N * np.cos(gamma_v)) / (self.m * V * np.cos(theta_v))

        d_ang_vel = np.zeros(3)
        d_ang_vel[0] = (self.M[0] - (self.J[2] - self.J[1]) * ang_vel[1] * ang_vel[2]) / self.J[0]
        d_ang_vel[1] = (self.M[1] - (self.J[0] - self.J[2]) * ang_vel[2] * ang_vel[0]) / self.J[1]
        d_ang_vel[2] = (self.M[2] - (self.J[1] - self.J[0]) * ang_vel[0] * ang_vel[1]) / self.J[2]

        d_theta = ang_vel[1] * np.sin(gamma) + ang_vel[2] * np.cos(gamma)
        d_phi = (ang_vel[1] * np.cos(gamma) - ang_vel[2] * np.sin(gamma)) / np.cos(theta)
        d_gamma = ang_vel[0] * - np.tan(theta) * (ang_vel[1] * np.cos(gamma) - ang_vel[2] * np.sin(gamma))

        d_pos = np.zeros(3)
        d_pos[0] = V * np.cos(theta_v) * np.cos(phi_v)
        d_pos[1] = V * np.sin(theta_v)
        d_pos[2] = -V * np.cos(theta_v) * np.sin(phi_v)

        d_vel = np.zeros(3)

        derivative = np.concatenate([d_pos, d_vel, d_ang_vel, [d_V, d_theta, d_phi, d_gamma, d_theta_v, d_phi_v, 0, 0, 0]])
        return derivative
    
    def Ny_controller(self, Nyc, Ny, wz, dt):
        # 过载跟踪误差
        self.eNy = Nyc - Ny

        # PI校正环节
        self.i_eNy += self.eNy * dt
        self.p_eNy = self.eNy

        pi_eNy = self.Kiz * self.i_eNy + self.Kpz * self.p_eNy

        # 增稳回路
        eSAC = pi_eNy - self.Kaz * wz
        self.i_eSAC += eSAC * dt

        # 阻尼回路
        eDamp = self.i_eSAC - self.Kwz * wz

        return eDamp

    def step(self, action):
        dt = action["dt"]
        Nyc = action["Nyc"]
        Vc = action["Vc"]

        self.delta_e = self.Ny_controller(Nyc, self.Ny, self.wz, dt*0.1)
        self.delta_e = np.clip(self.delta_e, -25 / 57.3, 25 / 57.3)
        
        # 计算气动力
        self._D()
        self._L()
        self._T()
        self._M()

        state = np.concatenate([self.pos, self.vel, self.ang_vel, [self.V, self.theta, self.phi, self.gamma, self.theta_v, self.phi_v, self.alpha, self.beta, self.gamma_v]])

        if self.integrator == "euler":
            derivative = self.d(state)
            state = state + derivative * dt
        elif self.integrator == "midpoint":
            temp1 = state + self.d(state) * (0.5 * dt)
            k1 = self.d(temp1)
            state = state + k1 * dt
        elif self.integrator == "rk23":
            k1 = self.d(state)
            temp1 = state + k1 * (0.5 * dt)
            k2 = self.d(temp1)
            temp2 = state + k2 * (0.5 * dt)
            k3 = self.d(temp2)
            state = state + (k1 + k2 * 2 + k3) * (dt / 4)
        elif self.integrator == "rk45":
            k1 = self.d(state)
            temp1 = state + k1 * (0.5 * dt)
            k2 = self.d(temp1)
            temp2 = state + k2 * (0.5 * dt)
            k3 = self.d(temp2)
            temp3 = state + k3 * dt
            k4 = self.d(temp3)
            state = state + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6)

        self.pos = state[:3]
        self.vel = state[3:6] 
        self.ang_vel = state[6:9]
        self.V, self.theta, self.phi, self.gamma, self.theta_v, self.phi_v, self.alpha, self.beta, self.gamma_v = state[9:]

        self.beta = np.cos(self.theta_v) * (np.cos(self.gamma) * np.sin(self.phi - self.phi_v) + np.sin(self.theta) * np.sin(self.gamma) * np.cos(self.phi - self.phi_v)) - np.sin(self.theta_v) * np.cos(self.theta) * np.sin(self.gamma)
        self.alpha = (np.cos(self.theta_v) * (np.sin(self.theta) * np.cos(self.gamma) * np.cos(self.phi - self.phi_v) - np.sin(self.gamma) * np.sin(self.phi - self.phi_v)) - np.sin(self.theta_v) * np.cos(self.theta) * np.cos(self.gamma)) / np.cos(self.beta)
        self.gamma_v = (np.cos(self.alpha) * np.sin(self.beta) * np.sin(self.theta) - np.sin(self.alpha) * np.sin(self.beta) * np.cos(self.gamma) * np.cos(self.theta) + np.cos(self.beta) * np.sin(self.gamma) * np.cos(self.theta)) / np.cos(self.theta_v)

        self.vel[0] = self.V * np.cos(self.theta_v) * np.cos(self.phi_v)
        self.vel[1] = self.V * np.sin(self.theta_v)
        self.vel[2] = -self.V * np.cos(self.theta_v) * np.sin(self.phi_v)
        self.h = self.pos[1]

        self.Tem = Temperature(self.h)
        self.Pres = Pressure(self.h)
        self.Rho = Density(self.Tem, self.Pres)
        self.a = SpeedofSound(self.Tem)
        self.g = Gravity(self.h)

        self.q = 0.5 * self.Rho * self.V * self.V
        self.Ny = (self.T * (np.sin(self.alpha) * np.cos(self.gamma_v) - np.cos(self.alpha) * np.sin(self.beta) * np.sin(self.gamma_v))
                                + self.L * np.cos(self.gamma_v) - self.N * np.sin(self.gamma_v) - self.m * self.g * np.cos(self.theta_v)) / (self.m * self.g)
        self.wz = self.ang_vel[2]

        return self.to_dict()

register("wingedcone_py-v0", "aerodrome.envs.WingedCone_py:WingedCone_py")