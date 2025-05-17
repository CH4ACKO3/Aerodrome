// This file contains code from EthanJamesLew/f16-flight-dynamics, licensed under GNU GPL-3.0.
// Original source: https://github.com/EthanJamesLew/f16-flight-dynamics
// Copyright (C) 2025 EthanJamesLew
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU GPL-3.0 along with this program.
// If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include <vector>
#include <string>
#include <array>
#include <cmath>

//
// Created by elew on 4/23/22.
//

#define SIGN_FACTOR 1.1f

struct adc_return {
    double amach;
    double qbar;
};

/* floor/ceil double based on sign and convert to integer */
inline int fix(double ele) {
  if (ele > 0.0)
    return int(floor(ele));
  else
    return int(ceil(ele));
}

/* sign of a number, return 0 if 0 */
inline int sign(double ele) {
  if (ele < 0.0)
    return -1;
  else if (ele == 0)
    return 0;
  else
    return 1;
}

/* rolling moment LUT */
double cl_a[12][7] = {{0.0f, -0.001f, -0.003f, -0.001f, 0.0f, 0.007f, 0.009f},
                      {0.0f, -0.004f, -0.009f, -0.01f, -0.01f, -0.01f, -0.011f},
                      {0.0f, -0.008f, -0.017f, -0.02f, -0.022f, -0.023f, -0.023f},
                      {0.0f, -0.012f, -0.024f, -0.03f, -0.034f, -0.034f, -0.037f},
                      {0.0f, -0.016f, -0.03f, -0.039f, -0.047f, -0.049f, -0.05f},
                      {0.0f, -0.022f, -0.041f, -0.054f, -0.06f, -0.063f, -0.068f},
                      {0.0f, -0.022f, -0.045f, -0.057f, -0.069f, -0.081f, -0.089f},
                      {0.0f, -0.021f, -0.04f, -0.054f, -0.067f, -0.079f, -0.088f},
                      {0.0f, -0.015f, -0.016f, -0.023f, -0.033f, -0.06f, -0.091f},
                      {0.0f, -0.008f, -0.002f, -0.006f, -0.036f, -0.058f, -0.076f},
                      {0.0f, -0.013f, -0.01f, -0.014f, -0.035f, -0.062f, -0.077f},
                      {0.0f, -0.015f, -0.019f, -0.027f, -0.035f, -0.059f, -0.076f}};

/* pitching moment LUT */
double cm_a[12][5] = {{0.205f, 0.081f, -0.046f, -0.174f, -0.259f},
                      {0.168f, 0.077f, -0.02f, -0.145f, -0.202f},
                      {0.186f, 0.107f, -0.009f, -0.121f, -0.184f},
                      {0.196f, 0.11f, -0.005f, -0.127f, -0.193f},
                      {0.213f, 0.11f, -0.006f, -0.129f, -0.199f},
                      {0.251f, 0.141f, 0.01f, -0.102f, -0.15f},
                      {0.245f, 0.127f, 0.006f, -0.097f, -0.16f},
                      {0.238f, 0.119f, -0.001f, -0.113f, -0.167f},
                      {0.252f, 0.133f, 0.014f, -0.087f, -0.104f},
                      {0.231f, 0.108f, 0.0f, -0.084f, -0.076f},
                      {0.198f, 0.081f, -0.013f, -0.069f, -0.041f},
                      {0.192f, 0.093f, 0.032f, -0.006f, -0.005f}};

/* yawing moment LUT */
double cn_a[12][7] = {{0.0f, 0.018f, 0.038f, 0.056f, 0.064f, 0.074f, 0.079f},
                      {0.0f, 0.019f, 0.042f, 0.057f, 0.077f, 0.086f, 0.09f},
                      {0.0f, 0.018f, 0.042f, 0.059f, 0.076f, 0.093f, 0.106f},
                      {0.0f, 0.019f, 0.042f, 0.058f, 0.074f, 0.089f, 0.106f},
                      {0.0f, 0.019f, 0.043f, 0.058f, 0.073f, 0.08f, 0.096f},
                      {0.0f, 0.018f, 0.039f, 0.053f, 0.057f, 0.062f, 0.08f},
                      {0.0f, 0.013f, 0.03f, 0.032f, 0.029f, 0.049f, 0.068f},
                      {0.0f, 0.007f, 0.017f, 0.012f, 0.007f, 0.022f, 0.03f},
                      {0.0f, 0.004f, 0.004f, 0.002f, 0.012f, 0.028f, 0.064f},
                      {0.0f, -0.014f, -0.035f, -0.046f, -0.034f, -0.012f, 0.015f},
                      {0.0f, -0.017f, -0.047f, -0.071f, -0.065f, -0.002f, 0.011f},
                      {0.0f, -0.033f, -0.057f, -0.073f, -0.041f, -0.013f, -0.001f}};

/* x-axis aerodynamic force coefficient LUT */
double cx_a[12][5] = {{-0.099f, -0.048f, -0.022f, -0.04f, -0.083f},
                      {-0.081f, -0.038f, -0.02f, -0.038f, -0.073f},
                      {-0.081f, -0.04f, -0.021f, -0.039f, -0.076f},
                      {-0.063f, -0.021f, -0.004f, -0.025f, -0.072f},
                      {-0.025f, 0.016f, 0.032f, 0.006f, -0.046f},
                      {0.044f, 0.083f, 0.094f, 0.062f, 0.012f},
                      {0.097f, 0.127f, 0.128f, 0.087f, 0.024f},
                      {0.113f, 0.137f, 0.13f, 0.085f, 0.025f},
                      {0.145f, 0.162f, 0.154f, 0.1f, 0.043f},
                      {0.167f, 0.177f, 0.161f, 0.11f, 0.053f},
                      {0.174f, 0.179f, 0.155f, 0.104f, 0.047f},
                      {0.166f, 0.167f, 0.138f, 0.091f, 0.04f}};
// MorelliParameters MV = MorelliParameters();


/* damping coefficients LUT */
double dampp_a[12][9] = {{-0.267f, 0.882f, -0.108f, -8.8f, -0.126f, -0.36f, -7.21f, -0.38f, 0.061f},
                         {-0.11f, 0.852f, -0.108f, -25.8f, -0.026f, -0.359f, -0.54f, -0.363f, 0.052f},
                         {0.308f, 0.876f, -0.188f, -28.9f, 0.063f, -0.443f, -5.23f, -0.378f, 0.052f},
                         {1.34f, 0.958f, 0.11f, -31.4f, 0.113f, -0.42f, -5.26f, -0.386f, -0.012f},
                         {2.08f, 0.962f, 0.258f, -31.2f, 0.208f, -0.383f, -6.11f, -0.37f, -0.013f},
                         {2.91f, 0.974f, 0.226f, -30.7f, 0.23f, -0.375f, -6.64f, -0.453f, -0.024f},
                         {2.76f, 0.819f, 0.344f, -27.7f, 0.319f, -0.329f, -5.69f, -0.55f, 0.05f},
                         {2.05f, 0.483f, 0.362f, -28.2f, 0.437f, -0.294f, -6.0f, -0.582f, 0.15f},
                         {1.5f, 0.59f, 0.611f, -29.0f, 0.68f, -0.23f, -6.2f, -0.595f, 0.13f},
                         {1.49f, 1.21f, 0.529f, -29.8f, 0.1f, -0.21f, -6.4f, -0.637f, 0.158f},
                         {1.83f, -0.493f, 0.298f, -38.3f, 0.447f, -0.12f, -6.6f, -1.02f, 0.24f},
                         {1.21f, -1.04f, -2.27f, -35.3f, -0.33f, -0.1f, -6.0f, -0.84f, 0.15f}};

/* rolling moment due to ailerons LUT */
double dlda_a[12][7] = {{-0.041f, -0.041f, -0.042f, -0.04f, -0.043f, -0.044f, -0.043f},
                        {-0.052f, -0.053f, -0.053f, -0.052f, -0.049f, -0.048f, -0.049f},
                        {-0.053f, -0.053f, -0.052f, -0.051f, -0.048f, -0.048f, -0.047f},
                        {-0.056f, -0.053f, -0.051f, -0.052f, -0.049f, -0.047f, -0.045f},
                        {-0.05f, -0.05f, -0.049f, -0.048f, -0.043f, -0.042f, -0.042f},
                        {-0.056f, -0.051f, -0.049f, -0.048f, -0.042f, -0.041f, -0.037f},
                        {-0.082f, -0.066f, -0.043f, -0.042f, -0.042f, -0.02f, -0.003f},
                        {-0.059f, -0.043f, -0.035f, -0.037f, -0.036f, -0.028f, -0.013f},
                        {-0.042f, -0.038f, -0.026f, -0.031f, -0.025f, -0.013f, -0.01f},
                        {-0.038f, -0.027f, -0.016f, -0.026f, -0.021f, -0.014f, -0.003f},
                        {-0.027f, -0.023f, -0.018f, -0.017f, -0.016f, -0.011f, -0.007f},
                        {-0.017f, -0.016f, -0.014f, -0.012f, -0.011f, -0.01f, -0.008f}};

/* rolling moment due to rudder LUT */
double dldr_a[12][7] = {{0.005f, 0.007f, 0.013f, 0.018f, 0.015f, 0.021f, 0.023f},
                        {0.017f, 0.016f, 0.013f, 0.015f, 0.014f, 0.011f, 0.01f},
                        {0.014f, 0.014f, 0.011f, 0.015f, 0.013f, 0.01f, 0.011f},
                        {0.01f, 0.014f, 0.012f, 0.014f, 0.013f, 0.011f, 0.011f},
                        {-0.005f, 0.013f, 0.011f, 0.014f, 0.012f, 0.01f, 0.011f},
                        {0.009f, 0.009f, 0.009f, 0.014f, 0.011f, 0.009f, 0.01f},
                        {0.019f, 0.012f, 0.008f, 0.014f, 0.011f, 0.008f, 0.008f},
                        {0.005f, 0.005f, 0.005f, 0.015f, 0.01f, 0.01f, 0.01f},
                        {-0.0f, 0.0f, -0.002f, 0.013f, 0.008f, 0.006f, 0.006f},
                        {-0.005f, 0.004f, 0.005f, 0.011f, 0.008f, 0.005f, 0.014f},
                        {-0.011f, 0.009f, 0.003f, 0.006f, 0.007f, 0.0f, 0.02f},
                        {0.008f, 0.007f, 0.005f, 0.001f, 0.003f, 0.001f, 0.0f}};

/* yawing moment due to ailerons LUT */
double dnda_a[12][7] = {{0.001f, 0.002f, -0.006f, -0.011f, -0.015f, -0.024f, -0.022f},
                        {-0.027f, -0.014f, -0.008f, -0.011f, -0.015f, -0.01f, 0.002f},
                        {-0.017f, -0.016f, -0.006f, -0.01f, -0.014f, -0.004f, -0.003f},
                        {-0.013f, -0.016f, -0.006f, -0.009f, -0.012f, -0.002f, -0.005f},
                        {-0.012f, -0.014f, -0.005f, -0.008f, -0.011f, -0.001f, -0.003f},
                        {-0.016f, -0.019f, -0.008f, -0.006f, -0.008f, 0.003f, -0.001f},
                        {0.001f, -0.021f, -0.005f, 0.0f, -0.002f, 0.014f, -0.009f},
                        {0.017f, 0.002f, 0.007f, 0.004f, 0.002f, 0.006f, -0.009f},
                        {0.011f, 0.012f, 0.004f, 0.007f, 0.006f, -0.001f, -0.001f},
                        {0.017f, 0.016f, 0.007f, 0.01f, 0.012f, 0.004f, 0.003f},
                        {0.008f, 0.015f, 0.006f, 0.004f, 0.011f, 0.004f, -0.002f},
                        {0.016f, 0.011f, 0.006f, 0.01f, 0.011f, 0.006f, 0.001f}};

/* yawing moment due to rudder LUT */
double dndr_a[12][7] = {{-0.018f, -0.028f, -0.037f, -0.048f, -0.043f, -0.052f, -0.062f},
                        {-0.052f, -0.051f, -0.041f, -0.045f, -0.044f, -0.034f, -0.034f},
                        {-0.052f, -0.043f, -0.038f, -0.045f, -0.041f, -0.036f, -0.027f},
                        {-0.052f, -0.046f, -0.04f, -0.045f, -0.041f, -0.036f, -0.028f},
                        {-0.054f, -0.045f, -0.04f, -0.044f, -0.04f, -0.035f, -0.027f},
                        {-0.049f, -0.049f, -0.038f, -0.045f, -0.038f, -0.028f, -0.027f},
                        {-0.059f, -0.057f, -0.037f, -0.047f, -0.034f, -0.024f, -0.023f},
                        {-0.051f, -0.052f, -0.03f, -0.048f, -0.035f, -0.023f, -0.023f},
                        {-0.03f, -0.03f, -0.027f, -0.049f, -0.035f, -0.02f, -0.019f},
                        {-0.037f, -0.033f, -0.024f, -0.045f, -0.029f, -0.016f, -0.009f},
                        {-0.026f, -0.03f, -0.019f, -0.033f, -0.022f, -0.01f, -0.025f},
                        {-0.013f, -0.008f, -0.013f, -0.016f, -0.009f, -0.014f, -0.01f}};

/* z-axis aerodynamic force coefficient LUT */
double
    cz_a[12] = {0.77f, 0.241f, -0.1f, -0.415f, -0.731f, -1.053f, -1.355f, -1.646f, -1.917f, -2.12f, -2.248f, -2.229f};

/* converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar) */
adc_return adc(double vt, double alt) {
  double tfac, t, rho, a, amach, qbar;
  const double ro = 2.377e-3f;

  tfac = 1.0f - 0.703e-5f * alt;

  t = alt >= 35000.f ? 390.f : 519.f * tfac;

  // rho = freestream mass density
  rho = ro * pow(tfac, 4.14f);

  // a = speed of sound at the ambient conditions
  // speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
  a = sqrt(1.4 * 1716.3 * t);

  // amach = mach number
  amach = vt / a;

  // qbar = dynamic pressure
  qbar = .5 * rho * vt * vt;

  return adc_return({amach, qbar});
}

/*
 * rolling moment coefficient Cl
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 719
 * taken from fortran code
 */
double cl(double alpha, double beta) {
  double s, da, db, t, u, v, w, dum;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = 0.2f * abs(beta);
  m = fix(s);
  if (m == 0)
    m = 1;
  if (m >= 6)
    m = 5;

  db = s - m;
  n = m + fix(SIGN_FACTOR * sign(db));
  l += 3;
  k += 3;
  m++;
  n++;

  t = cl_a[k - 1][m - 1];
  u = cl_a[k - 1][n - 1];
  v = t + abs(da) * (cl_a[l - 1][m - 1] - t);
  w = u + abs(da) * (cl_a[l - 1][n - 1] - u);
  dum = v + (w - v) * abs(db);

  return dum * sign(beta);
}

/*
 * pitching coefficient Cm
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 719
 * taken from fortran code
 */
double cm(double alpha, double el) {
  double s, de, da, t, u, v, w;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = el / 12.0f;
  m = fix(s);

  if (m <= -2)
    m = -1;
  if (m >= 2)
    m = 1;

  de = s - m;
  n = m + fix(SIGN_FACTOR * sign(de));
  l += 3;
  k += 3;
  m += 3;
  n += 3;

  t = cm_a[k - 1][m - 1];
  u = cm_a[k - 1][n - 1];
  v = t + abs(da) * (cm_a[l - 1][m - 1] - t);
  w = u + abs(da) * (cm_a[l - 1][n - 1] - u);

  return v + (w - v) * abs(de);
}

/*
 * yawing moment Cn
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 720
 * taken from fortran code
 */
double cn(double alpha, double beta) {
  double s, da, db, t, u, v, w, dum;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = 0.2f * abs(beta);
  m = fix(s);

  if (m == 0)
    m = 1;
  if (m >= 6)
    m = 5;

  db = s - m;
  n = m + fix(SIGN_FACTOR * sign(db));
  l += 3;
  k += 3;
  m++;
  n++;

  t = cn_a[k - 1][m - 1];
  u = cn_a[k - 1][n - 1];

  v = t + abs(da) * (cn_a[l - 1][m - 1] - t);
  w = u + abs(da) * (cn_a[l - 1][n - 1] - u);
  dum = v + (w - v) * abs(db);

  return dum * sign(beta);
}

/*
 * x-axis aerodynamic force coefficient Cx
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 718
 * taken from fortran code
 */
double cx(double alpha, double el) {
  double s, da, de, t, u, v, w, cxx;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = el / 12.0f;
  m = fix(s);

  if (m <= -2)
    m = -1;
  if (m >= 2)
    m = 1;

  de = s - m;
  n = m + fix(SIGN_FACTOR * sign(de));
  l += 3;
  k += 3;
  m += 3;
  n += 3;

  t = cx_a[k - 1][m - 1];
  u = cx_a[k - 1][n - 1];

  v = t + abs(da) * (cx_a[l - 1][m - 1] - t);
  w = u + abs(da) * (cx_a[l - 1][n - 1] - u);
  cxx = v + (w - v) * abs(de);

  return cxx;
}

/*
 * sideforce coefficient Cy
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 718
 * taken from fortran code
 */
double cy(double beta, double ail, double rdr) {
  return -0.02f * beta + 0.021f * (ail / 20.0f) + 0.086f * (rdr / 30.0f);
}

/*
 * z-axis aerodynamic force coefficient Cz
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 718
 * taken from fortran code
 */
double cz(double alpha, double beta, double el) {

  double s, da;
  int k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  l += 3;
  k += 3;
  s = cz_a[k - 1] + abs(da) * (cz_a[l - 1] - cz_a[k - 1]);

  return s * (1 - pow(beta / 57.3f, 2.0f) - 0.19f * (el / 25.0f));
}

/*
 * various damping coefficients
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 717
 * taken from fortran code
 */
std::array<double, 9> dampp(double alpha) {
  double s, da;
  int k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  l += 3;
  k += 3;

  /* store results in a boost array */
  std::array<double, 9> d = {0.0};
  for (size_t i = 0; i < d.size(); i++) {
    d[i] = dampp_a[k - 1][i] + abs(da) * (dampp_a[l - 1][i] - dampp_a[k - 1][i]);
  }

  return d;
}

/*
 * rolling mom. due to ailerons
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 720
 * taken from fortran code
 */
double dlda(double alpha, double beta) {
  double s, da, db, t, u, v, w;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = 0.1f * beta;
  m = fix(s);

  if (m <= -3)
    m = -2;
  if (m >= 3)
    m = 2;

  db = s - m;
  n = m + fix(SIGN_FACTOR * sign(db));
  l += 3;
  k += 3;
  m += 4;
  n += 4;

  t = dlda_a[k - 1][m - 1];
  u = dlda_a[k - 1][n - 1];

  v = t + abs(da) * (dlda_a[l - 1][m - 1] - t);
  w = u + abs(da) * (dlda_a[l - 1][n - 1] - u);
  return v + (w - v) * abs(db);

}

/*
 * rolling moment due to rudder
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 721
 * taken from fortran code
 */
double dldr(double alpha, double beta) {
  double s, da, db, t, u, v, w;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = 0.1f * beta;
  m = fix(s);

  if (m <= -3)
    m = -2;
  if (m >= 3)
    m = 2;

  db = s - m;
  n = m + fix(SIGN_FACTOR * sign(db));
  l += 3;
  k += 3;
  m += 4;
  n += 4;

  t = dldr_a[k - 1][m - 1];
  u = dldr_a[k - 1][n - 1];

  v = t + abs(da) * (dldr_a[l - 1][m - 1] - t);
  w = u + abs(da) * (dldr_a[l - 1][n - 1] - u);
  return v + (w - v) * abs(db);
}

/*
 * yawing moment due to ailerons
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 721
 * taken from fortran code
 */
double dnda(double alpha, double beta) {
  double s, da, db, t, u, v, w;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = 0.1f * beta;
  m = fix(s);

  if (m <= -3)
    m = -2;
  if (m >= 3)
    m = 2;

  db = s - m;
  n = m + fix(SIGN_FACTOR * sign(db));
  l += 3;
  k += 3;
  m += 4;
  n += 4;

  t = dnda_a[k - 1][m - 1];
  u = dnda_a[k - 1][n - 1];

  v = t + abs(da) * (dnda_a[l - 1][m - 1] - t);
  w = u + abs(da) * (dnda_a[l - 1][n - 1] - u);
  return v + (w - v) * abs(db);

}

/*
 * yawing moment due to rudder
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 721
 * taken from fortran code
 */
double dndr(double alpha, double beta) {
  double s, da, db, t, u, v, w;
  int m, n, k, l;

  s = 0.2f * alpha;
  k = fix(s);

  if (k <= -2)
    k = -1;
  if (k >= 9)
    k = 8;

  da = s - k;
  l = k + fix(SIGN_FACTOR * sign(da));
  s = 0.1f * beta;
  m = fix(s);

  if (m <= -3)
    m = -2;
  if (m >= 3)
    m = 2;

  db = s - m;
  n = m + fix(SIGN_FACTOR * sign(db));
  l += 3;
  k += 3;
  m += 4;
  n += 4;

  t = dndr_a[k - 1][m - 1];
  u = dndr_a[k - 1][n - 1];

  v = t + abs(da) * (dndr_a[l - 1][m - 1] - t);
  w = u + abs(da) * (dndr_a[l - 1][n - 1] - u);
  return v + (w - v) * abs(db);
}

/*
 * power vs throttle relationship
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 715
 * taken from fortran code
 */
double tgear(double thtl) {
  return thtl <= 0.77f ? 64.94f * thtl : 217.38f * thtl - 117.38f;
}

/* helper for pdot */
inline double rtau(double dp) {
  if (dp <= 25.0f) {
    return 1.0;
  } else if (dp >= 50.0f) {
    return 0.1f;
  } else {
    return 1.9f - 0.036f * dp;
  }
}

/*
 * rate of change of power
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 715
 * taken from fortran code
 */
double pdot(double p3, double p1) {
  double t, p2;

  if (p1 >= 50.0) {
    if (p3 >= 50) {
      t = 5.0;
      p2 = p1;
    } else {
      p2 = 60.0f;
      t = rtau(p2 - p3);
    }
  } else {
    if (p3 >= 50) {
      t = 5.0f;
      p2 = 40.0f;
    } else {
      p2 = p1;
      t = rtau(p2 - p3);
    }
  }
  return t * (p2 - p3);
}

double thrust_a[6][6] = {{1060.0f, 635.0f, 60.0f, -1020.0f, -2700.0f, -3600.0f},
                         {670.0f, 425.0f, 25.0f, -170.0f, -1900.0f, -1400.0f},
                         {880.0f, 690.0f, 345.0f, -300.0f, -1300.0f, -595.0f},
                         {1140.0f, 1010.0f, 755.0f, 350.0f, -247.0f, -342.0f},
                         {1500.0f, 1330.0f, 1130.0f, 910.0f, 600.0f, -200.0f},
                         {1860.0f, 1700.0f, 1525.0f, 1360.0f, 1100.0f, 700.0f}};

double thrust_b[6][6] = {{12680.0f, 12680.0f, 12610.0f, 12640.0f, 12390.0f, 11680.0f},
                         {9150.0f, 9150.0f, 9312.0f, 9839.0f, 10176.0f, 9848.0f},
                         {6200.0f, 6313.0f, 6610.0f, 7090.0f, 7750.0f, 8050.0f},
                         {3950.0f, 4040.0f, 4290.0f, 4660.0f, 5320.0f, 6100.0f},
                         {2450.0f, 2470.0f, 2600.0f, 2840.0f, 3250.0f, 3800.0f},
                         {1400.0f, 1400.0f, 1560.0f, 1660.0f, 1930.0f, 2310.0f}};

double thrust_c[6][6] = {{20000.0f, 21420.0f, 22700.0f, 24240.0f, 26070.0f, 28886.0f},
                         {15000.0f, 15700.0f, 16860.0f, 18910.0f, 21075.0f, 23319.0f},
                         {10800.0f, 11225.0f, 12250.0f, 13760.0f, 15975.0f, 18300.0f},
                         {7000.0f, 7323.0f, 8154.0f, 9285.0f, 11115.0f, 13484.0f},
                         {4000.0f, 4435.0f, 5000.0f, 5700.0f, 6860.0f, 8642.0f},
                         {2500.0f, 2600.0f, 2835.0f, 3215.0f, 3950.0f, 5057.0f}};

/*
 * engine thrust model
 * Stevens & Lewis, "Aircraft Control and Simulation", 3rd Ed pages 716
 * taken from fortran code
 */
double thrust(double power, double alt, double rmach) {
  double h, dh, dm, cdh, rm, s, t, tmil, tmax, thrst, tidl;
  int m, i;
  alt = alt < 0 ? 0.01f : alt;

  h = 0.0001f * alt;

  i = fix(h);

  if (i >= 5)
    i = 4;

  dh = h - i;
  rm = 5.0f * rmach;
  m = fix(rm);

  if (m >= 5)
    m = 4;
  else if (m <= 0)
    m = 0;

  dm = rm - m;
  cdh = 1 - dh;

  s = thrust_b[i][m] * cdh + thrust_b[i + 1][m] * dh;
  t = thrust_b[i][m + 1] * cdh + thrust_b[i + 1][m + 1] * dh;
  tmil = s + (t - s) * dm;

  if (power <= 50.0f) {
    s = thrust_a[i][m] * cdh + thrust_a[i + 1][m] * dh;
    t = thrust_a[i][m + 1] * cdh + thrust_a[i + 1][m + 1] * dh;
    tidl = s + (t - s) * dm;
    thrst = tidl + (tmil - tidl) * power * 0.02f;
  } else {
    s = thrust_c[i][m] * cdh + thrust_c[i + 1][m] * dh;
    t = thrust_c[i][m + 1] * cdh + thrust_c[i + 1][m + 1] * dh;
    tmax = s + (t - s) * dm;
    thrst = tmil + (tmax - tmil) * (power - 50.0f) * .02f;
  }
  return thrst;
}

// /*
//  * morelli model engine dynamics
//  */
Eigen::Matrix<double, 6, 1> morelli(double alpha,
                                double beta,
                                double de,
                                double da,
                                double dr,
                                double p,
                                double q,
                                double r,
                                double cbar,
                                double b,
                                double V,
                                double xcg,
                                double xcgref) {

  double  a0 = -1.943367e-2;
  double  a1 = 2.136104e-1;
  double  a2 = -2.903457e-1;
  double  a3 = -3.348641e-3;
  double  a4 = -2.060504e-1;
  double  a5 = 6.988016e-1;
  double  a6 = -9.035381e-1;

double b0 = 4.833383e-1;
double b1 = 8.644627;
double b2 = 1.131098e1;
double b3 = -7.422961e1;
double b4 = 6.075776e1;

double c0 = -1.145916;
double c1 = 6.016057e-2;
double c2 = 1.642479e-1;

double d0 = -1.006733e-1;
double d1 = 8.679799e-1;
double d2 = 4.260586;
double d3 = -6.923267;

double e0 = 8.071648e-1;
double e1 = 1.189633e-1;
double e2 = 4.177702;
double e3 = -9.162236;

double f0 = -1.378278e-1;
double f1 = -4.211369;
double f2 = 4.775187;
double f3 = -1.026225e1;
double f4 = 8.399763;
double f5 = -4.354000e-1;

double g0 = -3.054956e1;
double g1 = -4.132305e1;
double g2 = 3.292788e2;
double g3 = -6.848038e2;
double g4 = 4.080244e2;

double h0 = -1.05853e-1;
double h1 = -5.776677e-1;
double h2 = -1.672435e-2;
double h3 = 1.357256e-1;
double h4 = 2.172952e-1;
double h5 = 3.464156;
double h6 = -2.835451;
double h7 = -1.098104;

double i0 = -4.126806e-1;
double i1 = -1.189974e-1;
double i2 = 1.247721;
double i3 = -7.391132e-1;

double j0 = 6.250437e-2;
double j1 = 6.067723e-1;
double j2 = -1.101964;
double j3 = 9.100087;
double j4 = -1.192672e1;

double k0 = -1.463144e-1;
double k1 = -4.07391e-2;
double k2 = 3.253159e-2;
double k3 = 4.851209e-1;
double k4 = 2.978850e-1;
double k5 = -3.746393e-1;
double k6 = -3.213068e-1;

double l0 = 2.635729e-2;
double l1 = -2.192910e-2;
double l2 = -3.152901e-3;
double l3 = -5.817803e-2;
double l4 = 4.516159e-1;
double l5 = -4.928702e-1;
double l6 = -1.579864e-2;

double m0 = -2.029370e-2;
double m1 = 4.660702e-2;
double m2 = -6.012308e-1;
double m3 = -8.062977e-2;
double m4 = 8.320429e-2;
double m5 = 5.018538e-1;
double m6 = 6.378864e-1;
double m7 = 4.226356e-1;

double n0 = -5.19153;
double n1 = -3.554716;
double n2 = -3.598636e1;
double n3 = 2.247355e2;
double n4 = -4.120991e2;
double n5 = 2.411750e2;

double  o0 = 2.993363e-1;
double  o1 = 6.594004e-2;
double  o2 = -2.003125e-1;
double  o3 = -6.233977e-2;
double  o4 = -2.107885;
double  o5 = 2.141420;
double  o6 = 8.476901e-1;

double  p0 = 2.677652e-2;
double  p1 = -3.298246e-1;
double  p2 = 1.926178e-1;
double  p3 = 4.013325;
double  p4 = -4.404302;

double  q0 = -3.698756e-1;
double  q1 = -1.167551e-1;
double  q2 = -7.641297e-1;

double  r0 = -3.348717e-2;
double  r1 = 4.276655e-2;
double  r2 = 6.573646e-3;
double  r3 = 3.535831e-1;
double  r4 = -1.373308;
double  r5 = 1.237582;
double  r6 = 2.302543e-1;
double  r7 = -2.512876e-1;
double  r8 = 1.588105e-1;
double  r9 = -5.199526e-1;

double  s0 = -8.115894e-2;
double  s1 = -1.156580e-2;
double  s2 = 2.514167e-2;
double  s3 = 2.038748e-1;
double  s4 = -3.337476e-1;
double  s5 = 1.004297e-1;

  double phat, qhat, rhat;
  phat = p * b / (2 * V);
  qhat = q * cbar / (2 * V);
  rhat = r * b / (2 * V);

  double Cx0, Cxq, Cy0, Cyp, Cyr, Cz0, Czq, Cl0, Clp, Clr, Clda, Cldr, Cm0, Cmq, Cn0, Cnp, Cnr, Cnda, Cndr;
  Cx0 = a0 + a1 * alpha + a2 * pow(de, 2.0) + a3 * de + a4 * alpha * de + a5 * pow(alpha, 2.0)
      + a6 * pow(alpha, 3.0);
  Cxq = b0 + b1 * alpha + b2 * pow(alpha, 2.0) + b3 * pow(alpha, 3.0) + b4 * pow(alpha, 4.0);
  Cy0 = c0 * beta + c1 * da + c2 * dr;
  Cyp = d0 + d1 * alpha + d2 * pow(alpha, 2.0) + d3 * pow(alpha, 3.0);
  Cyr = e0 + e1 * alpha + e2 * pow(alpha, 2.0) + e3 * pow(alpha, 3.0);
  Cz0 = (f0 + f1 * alpha + f2 * pow(alpha, 2.0) + f3 * pow(alpha, 3.0) + f4 * pow(alpha, 4.0))
      * (1 - pow(beta, 2.0)) + f5 * de;
  Czq = g0 + g1 * alpha + g2 * pow(alpha, 2.0) + g3 * pow(alpha, 3.0) + g4 * pow(alpha, 4.0);
  Cl0 = h0 * beta + h1 * alpha * beta + h2 * pow(alpha, 2.0) * beta + h3 * pow(beta, 2.0)
      + h4 * alpha * pow(beta, 2.0) + h5 *
      pow(alpha, 3.0) * beta + h6 * pow(alpha, 4.0) * beta + h7 * pow(alpha, 2.0) * pow(beta, 2.0);
  Clp = i0 + i1 * alpha + i2 * pow(alpha, 2.0) + i3 * pow(alpha, 3.0);
  Clr = j0 + j1 * alpha + j2 * pow(alpha, 2.0) + j3 * pow(alpha, 3.0) + j4 * pow(alpha, 4.0);
  Clda = k0 + k1 * alpha + k2 * beta + k3 * pow(alpha, 2.0) + k4 * alpha * beta
      + k5 * pow(alpha, 2.0) * beta + k6 * pow(alpha, 3.0);
  Cldr = l0 + l1 * alpha + l2 * beta + l3 * alpha * beta + l4 * pow(alpha, 2.0) * beta
      + l5 * pow(alpha, 3.0) * beta + l6 * pow(beta, 2.0);
  Cm0 = m0 + m1 * alpha + m2 * de + m3 * alpha * de + m4 * pow(de, 2.0) + m5 * pow(alpha, 2.0) * de
      + m6 * pow(de, 3.0) + m7 *
      alpha * pow(de, 2.0);

  Cmq = n0 + n1 * alpha + n2 * pow(alpha, 2.0) + n3 * pow(alpha, 3.0) + n4 * pow(alpha, 4.0)
      + n5 * pow(alpha, 5.0);
  Cn0 = o0 * beta + o1 * alpha * beta + o2 * pow(beta, 2.0) + o3 * alpha * pow(beta, 2.0)
      + o4 * pow(alpha, 2.0) * beta + o5 *
      pow(alpha, 2.0) * pow(beta, 2.0) + o6 * pow(alpha, 3.0) * beta;
  Cnp = p0 + p1 * alpha + p2 * pow(alpha, 2.0) + p3 * pow(alpha, 3.0) + p4 * pow(alpha, 4.0);
  Cnr = q0 + q1 * alpha + q2 * pow(alpha, 2.0);
  Cnda = r0 + r1 * alpha + r2 * beta + r3 * alpha * beta + r4 * pow(alpha, 2.0) * beta
      + r5 * pow(alpha, 3.0) * beta + r6 *
      pow(alpha, 2.0) + r7 * pow(alpha, 3.0) + r8 * pow(beta, 3.0) + r9 * alpha * pow(beta, 3.0);
  Cndr = s0 + s1 * alpha + s2 * beta + s3 * alpha * beta + s4 * pow(alpha, 2.0) * beta
      + s5 * pow(alpha, 2.0);

  double Cx, Cy, Cz, Cl, Cm, Cn;
  Cx = Cx0 + Cxq * qhat;
  Cy = Cy0 + Cyp * phat + Cyr * rhat;
  Cz = Cz0 + Czq * qhat;
  Cl = Cl0 + Clp * phat + Clr * rhat + Clda * da + Cldr * dr;
  Cm = Cm0 + Cmq * qhat + Cz * (xcgref - xcg);
  Cn = Cn0 + Cnp * phat + Cnr * rhat + Cnda * da + Cndr * dr - Cy * (xcgref - xcg) * (cbar / b);

  Eigen::Matrix<double, 6, 1> result;
  result << Cx, Cy, Cz, Cl, Cm, Cn;

  return result;
}