#ifndef DENS_CITY_CORE_TYPES_H
#define DENS_CITY_CORE_TYPES_H

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace dens_city {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    constexpr Vec3() = default;
    constexpr Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    Vec3& operator*=(double s) { x *= s; y *= s; z *= s; return *this; }

    constexpr double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    double norm_sq() const { return dot(*this); }
    double norm() const { return std::sqrt(norm_sq()); }
};

struct Quaternion {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    constexpr Quaternion() = default;
    constexpr Quaternion(double w_, double x_, double y_, double z_) : w(w_), x(x_), y(y_), z(z_) {}

    static Quaternion random(double u1, double u2, double u3) {
        double r1 = std::sqrt(1.0 - u1);
        double r2 = std::sqrt(u1);
        double t1 = 2.0 * M_PI * u2;
        double t2 = 2.0 * M_PI * u3;
        return {r1 * std::sin(t1), r1 * std::cos(t1), r2 * std::sin(t2), r2 * std::cos(t2)};
    }

    Vec3 rotate(const Vec3& v) const {
        double t2 = w * x, t3 = w * y, t4 = w * z;
        double t5 = -x * x, t6 = x * y, t7 = x * z;
        double t8 = -y * y, t9 = y * z, t10 = -z * z;
        return {
            2.0 * ((t8 + t10) * v.x + (t6 - t4) * v.y + (t3 + t7) * v.z) + v.x,
            2.0 * ((t4 + t6) * v.x + (t5 + t10) * v.y + (t9 - t2) * v.z) + v.y,
            2.0 * ((t7 - t3) * v.x + (t2 + t9) * v.y + (t5 + t8) * v.z) + v.z
        };
    }
};

enum class MoleculeType : int {
    NONE = 0,
    SINGLE_SITE = 1,     // LJ, Hard Sphere, WCA, single ion
    TWO_TYPE = 2,        // Restricted Primitive Model (RPM) 1:1 electrolyte
    ABC_DIPOLE = 3,      // Rigid linear triatomic dipole (-q, 0, +q)
    WATER_3SITE = 4,     // Rigid 3-site water (SPC/E, TIP4P oxygen + hydrogens)
    CO2_3SITE = 5        // Rigid linear 3-site carbon dioxide (O=C=O, TraPPE / PBE-D3)
};

enum class ElectrostaticsMode : int {
    SHORT_RANGE = 0,     // Gaussian-truncated local reference (erfc(kappa*r)/r)
    LONG_RANGE_EWALD = 1 // Full 3D Ewald summation (real + reciprocal k-sum + self)
};

struct EwaldKVector {
    Vec3 k;
    double k_sq;
    double weight;
};

struct EwaldParams {
    double alpha = 0.35; // Screening parameter in 1/A
    int kmax = 4;        // Max integer wavevector index
    double self_energy_per_q2 = 0.0;
    std::vector<EwaldKVector> k_vectors;
};

struct ComplexDouble {
    double re = 0.0;
    double im = 0.0;
};

} // namespace dens_city

#endif // DENS_CITY_CORE_TYPES_H
