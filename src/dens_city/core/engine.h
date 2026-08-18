#ifndef DENS_CITY_ENGINE_H
#define DENS_CITY_ENGINE_H

#include "core_types.h"
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace dens_city {

class FastRNG {
private:
    uint64_t s[2];
    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }
public:
    explicit FastRNG(uint64_t seed = 42) {
        uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[0] = z ^ (z >> 31);
        z = (seed + 1) + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        s[1] = z ^ (z >> 31);
    }
    inline uint64_t next() {
        const uint64_t s0 = s[0];
        uint64_t s1 = s[1];
        const uint64_t result = rotl(s0 * 5, 7) * 9;
        s1 ^= s0;
        s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
        s[1] = rotl(s1, 37);
        return result;
    }
    inline double uniform() {
        return (next() >> 11) * 0x1.0p-53;
    }
    inline double uniform_range(double min_val, double max_val) {
        return min_val + (max_val - min_val) * uniform();
    }
    inline int randint(int min_val, int max_val) {
        if (min_val > max_val) return min_val;
        return min_val + static_cast<int>(next() % (max_val - min_val + 1));
    }
};

enum class PairPotentialKind : int {
    NONE = 0,
    LENNARD_JONES = 1,
    WCA = 2,
    HARD_SPHERE = 3,
    HARD_SPHERE_COULOMB = 4,
    LJ_COULOMB_GT = 5
};

struct PairPotential {
    PairPotentialKind kind = PairPotentialKind::NONE;
    double epsilon_lj = 0.0;
    double sigma_lj = 0.0;
    double rc = 0.0;
    double epsilon_c = 0.0;
    double q1 = 0.0, q2 = 0.0;
    double kappa_inv = 0.0;
    double diameter = 0.0;
    double prefactor = 1.0;
    double shift_lj = 0.0;

    double evaluate(double r) const;
};

enum class ExternalPotentialKind : int {
    NONE = 0,
    SLIT = 1,
    LJ93_WALL = 2,
    COSINE_CHARGE = 3
};

struct ExternalPotential {
    ExternalPotentialKind kind = ExternalPotentialKind::NONE;
    double low = 0.0, high = 0.0, width = 0.0, L = 0.0;
    double epsilon = 0.0, sigma = 0.0, cutoff = 0.0, shift = 0.0, q = 0.0;
    double A1 = 0.0, A2 = 0.0, A3 = 0.0, A4 = 0.0;
    double phi1 = 0.0, phi2 = 0.0, phi3 = 0.0, phi4 = 0.0;
    double q_A1 = 0.0, q_A2 = 0.0, q_A3 = 0.0, q_A4 = 0.0;
    double q_phi1 = 0.0, q_phi2 = 0.0, q_phi3 = 0.0, q_phi4 = 0.0;

    double evaluate(const Vec3& pos) const;
};

struct Molecule {
    std::vector<Vec3> sites;
    int species = 0;
};

class SimulationEngine {
public:
    double T = 300.0;
    double beta = 1.0;
    double mu1 = 0.0;
    double mu2 = 0.0;
    double kB = 1.380649e-23;

    double box_x = 20.0, box_y = 20.0, box_z = 20.0;
    double maxdispl = 0.5;
    double maxrot = 0.2;
    double bond_length = 1.0;

    int max_molecules = 2048;
    int current_step = 0;
    int max_steps = 10000;
    int equilibration = 1000;
    int output_interval = 1000;

    double prob_insert = 0.25;
    double prob_delete = 0.25;
    double prob_displace = 0.25;
    double prob_rotate = 0.25;
    double prob_mutate = 0.0;

    MoleculeType mol_type = MoleculeType::SINGLE_SITE;
    ElectrostaticsMode electrostatics_mode = ElectrostaticsMode::SHORT_RANGE;
    EwaldParams ewald_params;
    std::vector<ComplexDouble> rho_k; // Cached structure factor per wavevector k

    std::vector<Molecule> molecules;
    std::vector<std::vector<PairPotential>> pair_potentials; // [type_i][type_j]
    std::vector<ExternalPotential> external_potentials;      // [type_i]

    FastRNG rng;

    SimulationEngine();

    void init_ewald(double alpha, int kmax);
    void refresh_rho_k();
    double calc_ewald_reciprocal_energy_delta(const std::vector<ComplexDouble>& delta_rho_k) const;
    double calc_mol_self_energy(int species, int num_sites) const;
    void calc_mol_delta_rho_k(const Molecule& mol, double sign, std::vector<ComplexDouble>& delta_rho_k) const;

    void step();
    void step_single();
    void step_two_type();
    void step_abc();
    void step_h2o();
    void step_co2();

    double total_energy() const;
    int get_molecule_count() const { return static_cast<int>(molecules.size()); }
};

} // namespace dens_city

#endif // DENS_CITY_ENGINE_H
