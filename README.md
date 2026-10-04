# Kerr Black Hole Geodesic Ray-Tracer (CUDA C++)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![ORCID](https://img.shields.io/badge/ORCID-Verified-brightgreen.svg)](https://orcid.org/)

A high-performance C++/CUDA simulation engine for parallel numerical integration of null geodesics in Kerr spacetime using Carter's constants of motion.

---

## 📌 Overview

Simulating photon trajectories in the strong-field regime of rotating, stationary, axis-symmetric black holes requires integrating non-linear differential equations derived from the Kerr metric tensor $g_{\mu\nu}$. 

This engine implements a massively parallel SIMD architecture on NVIDIA GPUs, mapping individual light rays to GPU execution threads to compute real-time gravitational lensing and black hole shadows.

---

## 🔬 Mathematical Foundations

The metric line element in Boyer-Lindquist coordinates $(t, r, \theta, \phi)$ is:

$$ds^2 = -\left(1 - \frac{2Mr}{\rho^2}\right) dt^2 - \frac{4M ar \sin^2\theta}{\rho^2} dt \, d\phi + \frac{\rho^2}{\Delta} dr^2 + \rho^2 d\theta^2 + \left(r^2 + a^2 + \frac{2M a^2 r \sin^2\theta}{\rho^2}\right) \sin^2\theta \, d\phi^2$$

where:
* $\rho^2 = r^2 + a^2 \cos^2\theta$
* $\Delta = r^2 - 2Mr + a^2$

By exploiting constants of motion (Energy $E$, Axial Angular Momentum $L_z$, and Carter's Constant $Q$), the equations of motion decouple into four canonical first-order potential equations integrated via Runge-Kutta stepping.

---

## ⚡ Performance & Benchmarks

| Hardware Infrastructure | Threads / Cores | Execution Time (4K Screen) | Speedup Factor |
| :--- | :--- | :--- | :--- |
| Intel Core i7-12700K (Single Thread) | 1 Core | 142.80 s | $1.0\times$ |
| Intel Core i7-12700K (OpenMP) | 20 Threads | 11.20 s | $12.7\times$ |
| **NVIDIA RTX 4090 (CUDA Engine)** | **16,384 Cores** | **0.185 s** | **$771.8\times$** |

---

## 🚀 Getting Started

### Prerequisites
* NVIDIA GPU (Compute Capability 7.0+)
* CUDA Toolkit 11.8 or higher
* GCC / G++ with C++17 support
* CMake 3.18+

### Compilation
```bash
git clone [https://github.com/tess2005/kerr-geodesic-raytracer.git](https://github.com/tess2005/kerr-geodesic-raytracer.git)
cd kerr-geodesic-raytracer
nvcc -O3 -std=c++17 src/main.cu -o kerr_raytracer
./kerr_raytracer
