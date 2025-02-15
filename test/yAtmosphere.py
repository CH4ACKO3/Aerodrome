"""
U.S. STANDARD ATMOSPHERE 1976 implementation
Converted from original C++ code by YAGENAUT@GMAIL.COM
"""

import math

class YAtmosphere:
    # Transition points table [table 4]
    # Columns: [Altitude (m), Lapse rate (K/m), Temperature (K), Pressure (Pa)]
    TABLE4 = [
        [0,     -0.0065, 288.15, 1.01325e5],
        [11000,  0.0000, 216.65, 2.26320639734629e4],
        [20000,  0.0010, 216.65, 5.47488866967777e3],
        [32000,  0.0028, 228.65, 8.68018684755228e2],
        [47000,  0.0000, 270.65, 1.10906305554966e2],
        [51000, -0.0028, 270.65, 6.69388731186873e1],
        [71000, -0.0020, 214.65, 3.95642042804073e0],
        [84852,  0.0000, 186.946, 3.73383589976215e-1]
    ]

    @staticmethod
    def temperature(z):
        """Calculate temperature (K) at given altitude (m)"""
        H = z * 6356766 / (z + 6356766)  # Geopotential height
        b = 0
        while b < 7 and H >= YAtmosphere.TABLE4[b+1][0]:
            b += 1
        return YAtmosphere.TABLE4[b][2] + YAtmosphere.TABLE4[b][1] * (H - YAtmosphere.TABLE4[b][0])

    @staticmethod
    def pressure(z):
        """Calculate pressure (Pa) at given altitude (m)"""
        H = z * 6356766 / (z + 6356766)
        b = 0
        while b < 7 and H >= YAtmosphere.TABLE4[b+1][0]:
            b += 1
            
        Hb, Lb, Tb, Pb = YAtmosphere.TABLE4[b]
        C = -0.0341631947363104  # -G0*M0/RSTAR
        
        if abs(Lb) > 1e-12:
            return Pb * math.pow(1 + Lb/Tb * (H - Hb), C/Lb)
        else:
            return Pb * math.exp(C * (H - Hb) / Tb)

    @staticmethod
    def density(T, P):
        """Calculate density (kg/m³) using temperature (K) and pressure (Pa)"""
        return P * 0.00348367635597379 / T

    @staticmethod
    def speed_of_sound(T):
        """Calculate speed of sound (m/s) using temperature (K)"""
        return math.sqrt(401.87430086589 * T)

    @staticmethod
    def gravity(z):
        """Calculate gravitational acceleration (m/s²) at given altitude (m)"""
        return 9.80665 * math.pow(1 + z/6356766, -2)



import matplotlib.pyplot as plt
import numpy as np

h = np.linspace(0, 80000, 1000)
T = np.array([YAtmosphere.temperature(z) for z in h])
P = np.array([YAtmosphere.pressure(z) for z in h])
rho = np.array([YAtmosphere.density(T[i], P[i]) for i in range(len(h))])
a = np.array([YAtmosphere.speed_of_sound(T[i]) for i in range(len(h))])
g = np.array([YAtmosphere.gravity(z) for z in h])

fig, ax = plt.subplots(2, 2)
ax[0, 0].plot(T, h)
ax[0, 0].set_title('Temperature')
ax[0, 0].set_xlabel('Temperature (K)')
ax[0, 0].set_ylabel('Altitude (m)')

ax[0, 1].plot(P, h)
ax[0, 1].set_title('Pressure')
ax[0, 1].set_xlabel('Pressure (Pa)')
ax[0, 1].set_ylabel('Altitude (m)')

ax[1, 0].plot(rho, h)
ax[1, 0].set_title('Density')
ax[1, 0].set_xlabel('Density (kg/m^3)')
ax[1, 0].set_ylabel('Altitude (m)')

ax[1, 1].plot(a, h)
ax[1, 1].set_title('Speed of sound')
ax[1, 1].set_xlabel('Speed of sound (m/s)')
ax[1, 1].set_ylabel('Altitude (m)')

plt.show()
