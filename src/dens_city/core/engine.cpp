#include "engine.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dens_city {

static constexpr double COULOMB_PREFACTOR = 1.67101e-19; // J * A / e^2

double PairPotential::evaluate(double r) const {
    if (r >= rc) return 0.0;
    if (r <= 1e-12) return 1e12;

    switch (kind) {
        case PairPotentialKind::LENNARD_JONES: {
            double s_r = sigma_lj / r;
            double s_r6 = s_r * s_r * s_r * s_r * s_r * s_r;
            return 4.0 * epsilon_lj * (s_r6 * s_r6 - s_r6) - shift_lj;
        }
        case PairPotentialKind::WCA: {
            double r_cut = std::pow(2.0, 1.0 / 6.0) * sigma_lj;
            if (r < r_cut) {
                double s_r = sigma_lj / r;
                double s_r6 = s_r * s_r * s_r * s_r * s_r * s_r;
                return 4.0 * epsilon_lj * (s_r6 * s_r6 - s_r6) + epsilon_lj;
            }
            return 0.0;
        }
        case PairPotentialKind::HARD_SPHERE: {
            if (r < diameter) return 1e12;
            return 0.0;
        }
        case PairPotentialKind::HARD_SPHERE_COULOMB: {
            if (r < diameter) return 1e12;
            double u_c = (prefactor * q1 * q2 / r) * std::erfc(r / kappa_inv);
            return u_c;
        }
        case PairPotentialKind::LJ_COULOMB_GT: {
            double u_lj = 0.0;
            if (r < rc) {
                double s_r = sigma_lj / r;
                double s_r6 = s_r * s_r * s_r * s_r * s_r * s_r;
                u_lj = 4.0 * epsilon_lj * (s_r6 * s_r6 - s_r6) - shift_lj;
            }
            double u_c = 0.0;
            if (std::abs(q1) > 1e-6 && std::abs(q2) > 1e-6) {
                u_c = (prefactor * q1 * q2 / r) * std::erfc(r / kappa_inv);
            }
            return u_lj + u_c;
        }
        default:
            return 0.0;
    }
}

double ExternalPotential::evaluate(const Vec3& pos) const {
    switch (kind) {
        case ExternalPotentialKind::SLIT: {
            if (pos.z < low || pos.z > high) return 1e12;
            return 0.0;
        }
        case ExternalPotentialKind::LJ93_WALL: {
            double z1 = pos.z;
            double z2 = L - pos.z;
            double u = 0.0;
            if (z1 > 0.0 && z1 < cutoff) {
                double s_z = sigma / z1;
                double s_z3 = s_z * s_z * s_z;
                u += (2.0 * M_PI * epsilon * sigma * sigma * sigma / 3.0) * (2.0 / 15.0 * s_z3 * s_z3 * s_z3 - s_z3) - shift;
            }
            if (z2 > 0.0 && z2 < cutoff) {
                double s_z = sigma / z2;
                double s_z3 = s_z * s_z * s_z;
                u += (2.0 * M_PI * epsilon * sigma * sigma * sigma / 3.0) * (2.0 / 15.0 * s_z3 * s_z3 * s_z3 - s_z3) - shift;
            }
            return u;
        }
        case ExternalPotentialKind::COSINE_CHARGE: {
            if (pos.z < low || pos.z > high) return 1e12;
            double u = 0.0;
            if (std::abs(q) > 1e-6 && L > 0.0) {
                if (std::abs(A1) > 1e-6) u += q * A1 * std::cos(2.0 * M_PI * 1.0 * pos.z / L + phi1);
                if (std::abs(A2) > 1e-6) u += q * A2 * std::cos(2.0 * M_PI * 2.0 * pos.z / L + phi2);
                if (std::abs(A3) > 1e-6) u += q * A3 * std::cos(2.0 * M_PI * 3.0 * pos.z / L + phi3);
                if (std::abs(A4) > 1e-6) u += q * A4 * std::cos(2.0 * M_PI * 4.0 * pos.z / L + phi4);
            }
            return u;
        }
        default:
            return 0.0;
    }
}

SimulationEngine::SimulationEngine() {
    molecules.reserve(max_molecules);
    pair_potentials.resize(4, std::vector<PairPotential>(4));
    external_potentials.resize(4);
}

void SimulationEngine::init_ewald(double alpha, int kmax) {
    ewald_params.alpha = alpha;
    ewald_params.kmax = kmax;
    ewald_params.k_vectors.clear();

    double volume = box_x * box_y * box_z;
    double two_pi_lx = 2.0 * M_PI / box_x;
    double two_pi_ly = 2.0 * M_PI / box_y;
    double two_pi_lz = 2.0 * M_PI / box_z;

    for (int nx = -kmax; nx <= kmax; ++nx) {
        for (int ny = -kmax; ny <= kmax; ++ny) {
            for (int nz = -kmax; nz <= kmax; ++nz) {
                if (nx == 0 && ny == 0 && nz == 0) continue;
                if (nx * nx + ny * ny + nz * nz > kmax * kmax) continue;

                Vec3 k_vec(nx * two_pi_lx, ny * two_pi_ly, nz * two_pi_lz);
                double k_sq = k_vec.norm_sq();
                double weight = (4.0 * M_PI / (volume * k_sq)) * std::exp(-k_sq / (4.0 * alpha * alpha));
                weight *= COULOMB_PREFACTOR;

                ewald_params.k_vectors.push_back({k_vec, k_sq, weight});
            }
        }
    }

    ewald_params.self_energy_per_q2 = (alpha / std::sqrt(M_PI)) * COULOMB_PREFACTOR;
    rho_k.resize(ewald_params.k_vectors.size(), {0.0, 0.0});
    refresh_rho_k();
}

void SimulationEngine::refresh_rho_k() {
    if (ewald_params.k_vectors.empty()) return;
    std::fill(rho_k.begin(), rho_k.end(), ComplexDouble{0.0, 0.0});

    for (const auto& mol : molecules) {
        int num_sites = static_cast<int>(mol.sites.size());
        for (size_t k = 0; k < ewald_params.k_vectors.size(); ++k) {
            const auto& kv = ewald_params.k_vectors[k];
            for (int s = 0; s < num_sites; ++s) {
                double q = 0.0;
                if (mol_type == MoleculeType::ABC_DIPOLE) {
                    if (s == 0) q = -0.382;
                    else if (s == 2) q = +0.382;
                } else if (mol_type == MoleculeType::WATER_3SITE) {
                    if (s == 0) q = -0.8476;
                    else q = +0.4238;
                } else if (mol_type == MoleculeType::TWO_TYPE) {
                    q = (mol.species == 0) ? +1.0 : -1.0;
                } else if (mol_type == MoleculeType::CO2_3SITE) {
                    if (s == 0) q = +0.70;
                    else q = -0.35;
                }
                if (std::abs(q) < 1e-6) continue;
                double k_dot_r = kv.k.dot(mol.sites[s]);
                rho_k[k].re += q * std::cos(k_dot_r);
                rho_k[k].im += q * std::sin(k_dot_r);
            }
        }
    }
}

void SimulationEngine::calc_mol_delta_rho_k(const Molecule& mol, double sign, std::vector<ComplexDouble>& delta_rho_k) const {
    delta_rho_k.assign(ewald_params.k_vectors.size(), {0.0, 0.0});
    int num_sites = static_cast<int>(mol.sites.size());

    for (size_t k = 0; k < ewald_params.k_vectors.size(); ++k) {
        const auto& kv = ewald_params.k_vectors[k];
        double dre = 0.0, dim = 0.0;
        for (int s = 0; s < num_sites; ++s) {
            double q = 0.0;
            if (mol_type == MoleculeType::ABC_DIPOLE) {
                if (s == 0) q = -0.382;
                else if (s == 2) q = +0.382;
            } else if (mol_type == MoleculeType::WATER_3SITE) {
                if (s == 0) q = -0.8476;
                else q = +0.4238;
            } else if (mol_type == MoleculeType::TWO_TYPE) {
                q = (mol.species == 0) ? +1.0 : -1.0;
            } else if (mol_type == MoleculeType::CO2_3SITE) {
                if (s == 0) q = +0.70;
                else q = -0.35;
            }
            if (std::abs(q) < 1e-6) continue;
            double k_dot_r = kv.k.dot(mol.sites[s]);
            dre += sign * q * std::cos(k_dot_r);
            dim += sign * q * std::sin(k_dot_r);
        }
        delta_rho_k[k] = {dre, dim};
    }
}

double SimulationEngine::calc_ewald_reciprocal_energy_delta(const std::vector<ComplexDouble>& delta_rho_k) const {
    double delta_E = 0.0;
    for (size_t k = 0; k < ewald_params.k_vectors.size(); ++k) {
        double w = ewald_params.k_vectors[k].weight;
        double re = rho_k[k].re;
        double im = rho_k[k].im;
        double dre = delta_rho_k[k].re;
        double dim = delta_rho_k[k].im;
        delta_E += 0.5 * w * ((re + dre) * (re + dre) + (im + dim) * (im + dim) - (re * re + im * im));
    }
    return delta_E;
}

double SimulationEngine::calc_mol_self_energy(int species, int num_sites) const {
    double q_sq_sum = 0.0;
    if (mol_type == MoleculeType::ABC_DIPOLE) {
        q_sq_sum = 2.0 * (0.382 * 0.382);
    } else if (mol_type == MoleculeType::WATER_3SITE) {
        q_sq_sum = (-0.8476 * -0.8476) + 2.0 * (0.4238 * 0.4238);
    } else if (mol_type == MoleculeType::TWO_TYPE) {
        q_sq_sum = 1.0;
    } else if (mol_type == MoleculeType::CO2_3SITE) {
        q_sq_sum = (0.70 * 0.70) + 2.0 * (-0.35 * -0.35);
    }
    return -ewald_params.self_energy_per_q2 * q_sq_sum;
}

void SimulationEngine::step() {
    switch (mol_type) {
        case MoleculeType::SINGLE_SITE: step_single(); break;
        case MoleculeType::TWO_TYPE:    step_two_type(); break;
        case MoleculeType::ABC_DIPOLE:  step_abc(); break;
        case MoleculeType::WATER_3SITE: step_h2o(); break;
        case MoleculeType::CO2_3SITE:   step_co2(); break;
        default: break;
    }
    current_step++;
}

void SimulationEngine::step_single() {
    double r_move = rng.uniform();
    double volume = box_x * box_y * box_z;

    if (r_move < prob_insert) {
        if (static_cast<int>(molecules.size()) >= max_molecules) return;
        Vec3 pos(rng.uniform_range(0, box_x), rng.uniform_range(0, box_y), rng.uniform_range(0, box_z));
        Molecule new_mol;
        new_mol.sites.push_back(pos);
        new_mol.species = 0;

        double delta_E = external_potentials[0].evaluate(pos);
        if (delta_E > 1e10) return;

        for (const auto& other : molecules) {
            double dx = pos.x - other.sites[0].x;
            double dy = pos.y - other.sites[0].y;
            double dz = pos.z - other.sites[0].z;
            dx -= box_x * std::round(dx / box_x);
            dy -= box_y * std::round(dy / box_y);
            dz -= box_z * std::round(dz / box_z);
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            delta_E += pair_potentials[0][0].evaluate(r);
            if (delta_E > 1e10) return;
        }

        double log_p = -beta * (delta_E - mu1) + std::log(volume) - std::log(molecules.size() + 1);
        double prob = (log_p < 80.0) ? std::exp(log_p) : 0.0;
        if (rng.uniform() < prob) molecules.push_back(new_mol);
    } else if (r_move < prob_insert + prob_delete && !molecules.empty()) {
        int idx = rng.randint(0, static_cast<int>(molecules.size()) - 1);
        Vec3 pos = molecules[idx].sites[0];

        double delta_E = -external_potentials[0].evaluate(pos);
        for (size_t i = 0; i < molecules.size(); ++i) {
            if (static_cast<int>(i) == idx) continue;
            double dx = pos.x - molecules[i].sites[0].x;
            double dy = pos.y - molecules[i].sites[0].y;
            double dz = pos.z - molecules[i].sites[0].z;
            dx -= box_x * std::round(dx / box_x);
            dy -= box_y * std::round(dy / box_y);
            dz -= box_z * std::round(dz / box_z);
            double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            delta_E -= pair_potentials[0][0].evaluate(r);
        }

        double log_p = -beta * (delta_E + mu1) + std::log(molecules.size()) - std::log(volume);
        double prob = (log_p < 80.0) ? std::exp(log_p) : 0.0;
        if (rng.uniform() < prob) {
            molecules[idx] = molecules.back();
            molecules.pop_back();
        }
    } else if (!molecules.empty()) {
        int idx = rng.randint(0, static_cast<int>(molecules.size()) - 1);
        Vec3 old_pos = molecules[idx].sites[0];
        Vec3 new_pos(
            old_pos.x + rng.uniform_range(-maxdispl, maxdispl),
            old_pos.y + rng.uniform_range(-maxdispl, maxdispl),
            old_pos.z + rng.uniform_range(-maxdispl, maxdispl)
        );
        new_pos.x -= box_x * std::floor(new_pos.x / box_x);
        new_pos.y -= box_y * std::floor(new_pos.y / box_y);
        new_pos.z -= box_z * std::floor(new_pos.z / box_z);

        double delta_E = external_potentials[0].evaluate(new_pos) - external_potentials[0].evaluate(old_pos);
        if (delta_E < 1e10) {
            for (size_t i = 0; i < molecules.size(); ++i) {
                if (static_cast<int>(i) == idx) continue;
                double dx_old = old_pos.x - molecules[i].sites[0].x;
                double dy_old = old_pos.y - molecules[i].sites[0].y;
                double dz_old = old_pos.z - molecules[i].sites[0].z;
                dx_old -= box_x * std::round(dx_old / box_x);
                dy_old -= box_y * std::round(dy_old / box_y);
                dz_old -= box_z * std::round(dz_old / box_z);
                double r_old = std::sqrt(dx_old * dx_old + dy_old * dy_old + dz_old * dz_old);

                double dx_new = new_pos.x - molecules[i].sites[0].x;
                double dy_new = new_pos.y - molecules[i].sites[0].y;
                double dz_new = new_pos.z - molecules[i].sites[0].z;
                dx_new -= box_x * std::round(dx_new / box_x);
                dy_new -= box_y * std::round(dy_new / box_y);
                dz_new -= box_z * std::round(dz_new / box_z);
                double r_new = std::sqrt(dx_new * dx_new + dy_new * dy_new + dz_new * dz_new);

                delta_E += pair_potentials[0][0].evaluate(r_new) - pair_potentials[0][0].evaluate(r_old);
                if (delta_E > 1e10) break;
            }
        }
        if (delta_E < 1e10 && (delta_E <= 0.0 || rng.uniform() < std::exp(-beta * delta_E))) {
            molecules[idx].sites[0] = new_pos;
        }
    }
}

void SimulationEngine::step_two_type() {
    double r_move = rng.uniform();
    double volume = box_x * box_y * box_z;

    int count_pos = 0, count_neg = 0;
    for (const auto& m : molecules) {
        if (m.species == 0) count_pos++;
        else count_neg++;
    }

    if (r_move < prob_insert) {
        // Neutral pair insertion (cation + anion)
        if (static_cast<int>(molecules.size()) + 2 > max_molecules) return;
        Vec3 pos_p(rng.uniform_range(0, box_x), rng.uniform_range(0, box_y), rng.uniform_range(0, box_z));
        Vec3 pos_n(rng.uniform_range(0, box_x), rng.uniform_range(0, box_y), rng.uniform_range(0, box_z));

        Molecule mol_p, mol_n;
        mol_p.sites.push_back(pos_p); mol_p.species = 0;
        mol_n.sites.push_back(pos_n); mol_n.species = 1;

        double delta_E = external_potentials[0].evaluate(pos_p) + external_potentials[1].evaluate(pos_n);
        double dx_pn = pos_p.x - pos_n.x;
        double dy_pn = pos_p.y - pos_n.y;
        double dz_pn = pos_p.z - pos_n.z;
        dx_pn -= box_x * std::round(dx_pn / box_x);
        dy_pn -= box_y * std::round(dy_pn / box_y);
        dz_pn -= box_z * std::round(dz_pn / box_z);
        delta_E += pair_potentials[0][1].evaluate(std::sqrt(dx_pn * dx_pn + dy_pn * dy_pn + dz_pn * dz_pn));

        for (const auto& other : molecules) {
            double dx_p = pos_p.x - other.sites[0].x;
            double dy_p = pos_p.y - other.sites[0].y;
            double dz_p = pos_p.z - other.sites[0].z;
            dx_p -= box_x * std::round(dx_p / box_x);
            dy_p -= box_y * std::round(dy_p / box_y);
            dz_p -= box_z * std::round(dz_p / box_z);
            delta_E += pair_potentials[0][other.species].evaluate(std::sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p));

            double dx_n = pos_n.x - other.sites[0].x;
            double dy_n = pos_n.y - other.sites[0].y;
            double dz_n = pos_n.z - other.sites[0].z;
            dx_n -= box_x * std::round(dx_n / box_x);
            dy_n -= box_y * std::round(dy_n / box_y);
            dz_n -= box_z * std::round(dz_n / box_z);
            delta_E += pair_potentials[1][other.species].evaluate(std::sqrt(dx_n * dx_n + dy_n * dy_n + dz_n * dz_n));
            if (delta_E > 1e10) return;
        }

        std::vector<ComplexDouble> delta_k_p, delta_k_n;
        if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            calc_mol_delta_rho_k(mol_p, +1.0, delta_k_p);
            calc_mol_delta_rho_k(mol_n, +1.0, delta_k_n);
            std::vector<ComplexDouble> delta_k(ewald_params.k_vectors.size());
            for (size_t k = 0; k < delta_k.size(); ++k) {
                delta_k[k] = {delta_k_p[k].re + delta_k_n[k].re, delta_k_p[k].im + delta_k_n[k].im};
            }
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            delta_E += calc_mol_self_energy(0, 1) + calc_mol_self_energy(1, 1);
        }

        double log_p = -beta * (delta_E - (mu1 + mu2)) + 2.0 * std::log(volume) - std::log(count_pos + 1) - std::log(count_neg + 1);
        double prob = (log_p < 80.0) ? std::exp(log_p) : 0.0;
        if (rng.uniform() < prob) {
            molecules.push_back(mol_p);
            molecules.push_back(mol_n);
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < rho_k.size(); ++k) {
                    rho_k[k].re += delta_k_p[k].re + delta_k_n[k].re;
                    rho_k[k].im += delta_k_p[k].im + delta_k_n[k].im;
                }
            }
        }
    } else if (r_move < prob_insert + prob_delete && count_pos > 0 && count_neg > 0) {
        // Neutral pair deletion
        int idx_p = -1, idx_n = -1;
        int target_p = rng.randint(1, count_pos);
        int target_n = rng.randint(1, count_neg);
        int cur_p = 0, cur_n = 0;
        for (size_t i = 0; i < molecules.size(); ++i) {
            if (molecules[i].species == 0 && ++cur_p == target_p) idx_p = static_cast<int>(i);
            if (molecules[i].species == 1 && ++cur_n == target_n) idx_n = static_cast<int>(i);
        }
        if (idx_p < 0 || idx_n < 0 || idx_p == idx_n) return;

        Vec3 pos_p = molecules[idx_p].sites[0];
        Vec3 pos_n = molecules[idx_n].sites[0];

        double delta_E = -external_potentials[0].evaluate(pos_p) - external_potentials[1].evaluate(pos_n);
        double dx_pn = pos_p.x - pos_n.x;
        double dy_pn = pos_p.y - pos_n.y;
        double dz_pn = pos_p.z - pos_n.z;
        dx_pn -= box_x * std::round(dx_pn / box_x);
        dy_pn -= box_y * std::round(dy_pn / box_y);
        dz_pn -= box_z * std::round(dz_pn / box_z);
        delta_E -= pair_potentials[0][1].evaluate(std::sqrt(dx_pn * dx_pn + dy_pn * dy_pn + dz_pn * dz_pn));

        for (size_t i = 0; i < molecules.size(); ++i) {
            if (static_cast<int>(i) == idx_p || static_cast<int>(i) == idx_n) continue;
            double dx_p = pos_p.x - molecules[i].sites[0].x;
            double dy_p = pos_p.y - molecules[i].sites[0].y;
            double dz_p = pos_p.z - molecules[i].sites[0].z;
            dx_p -= box_x * std::round(dx_p / box_x);
            dy_p -= box_y * std::round(dy_p / box_y);
            dz_p -= box_z * std::round(dz_p / box_z);
            delta_E -= pair_potentials[0][molecules[i].species].evaluate(std::sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p));

            double dx_n = pos_n.x - molecules[i].sites[0].x;
            double dy_n = pos_n.y - molecules[i].sites[0].y;
            double dz_n = pos_n.z - molecules[i].sites[0].z;
            dx_n -= box_x * std::round(dx_n / box_x);
            dy_n -= box_y * std::round(dy_n / box_y);
            dz_n -= box_z * std::round(dz_n / box_z);
            delta_E -= pair_potentials[1][molecules[i].species].evaluate(std::sqrt(dx_n * dx_n + dy_n * dy_n + dz_n * dz_n));
        }

        std::vector<ComplexDouble> delta_k_p, delta_k_n;
        if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            calc_mol_delta_rho_k(molecules[idx_p], -1.0, delta_k_p);
            calc_mol_delta_rho_k(molecules[idx_n], -1.0, delta_k_n);
            std::vector<ComplexDouble> delta_k(ewald_params.k_vectors.size());
            for (size_t k = 0; k < delta_k.size(); ++k) {
                delta_k[k] = {delta_k_p[k].re + delta_k_n[k].re, delta_k_p[k].im + delta_k_n[k].im};
            }
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            delta_E -= (calc_mol_self_energy(0, 1) + calc_mol_self_energy(1, 1));
        }

        double log_p = -beta * (delta_E + (mu1 + mu2)) + std::log(count_pos) + std::log(count_neg) - 2.0 * std::log(volume);
        double prob = (log_p < 80.0) ? std::exp(log_p) : 0.0;
        if (rng.uniform() < prob) {
            int first = std::max(idx_p, idx_n);
            int second = std::min(idx_p, idx_n);
            molecules.erase(molecules.begin() + first);
            molecules.erase(molecules.begin() + second);
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < rho_k.size(); ++k) {
                    rho_k[k].re += delta_k_p[k].re + delta_k_n[k].re;
                    rho_k[k].im += delta_k_p[k].im + delta_k_n[k].im;
                }
            }
        }
    } else if (!molecules.empty()) {
        // Displace random ion
        int idx = rng.randint(0, static_cast<int>(molecules.size()) - 1);
        Vec3 old_pos = molecules[idx].sites[0];
        int sp = molecules[idx].species;
        Vec3 new_pos(
            old_pos.x + rng.uniform_range(-maxdispl, maxdispl),
            old_pos.y + rng.uniform_range(-maxdispl, maxdispl),
            old_pos.z + rng.uniform_range(-maxdispl, maxdispl)
        );
        new_pos.x -= box_x * std::floor(new_pos.x / box_x);
        new_pos.y -= box_y * std::floor(new_pos.y / box_y);
        new_pos.z -= box_z * std::floor(new_pos.z / box_z);

        double delta_E = external_potentials[sp].evaluate(new_pos) - external_potentials[sp].evaluate(old_pos);
        if (delta_E < 1e10) {
            for (size_t i = 0; i < molecules.size(); ++i) {
                if (static_cast<int>(i) == idx) continue;
                double dx_o = old_pos.x - molecules[i].sites[0].x;
                double dy_o = old_pos.y - molecules[i].sites[0].y;
                double dz_o = old_pos.z - molecules[i].sites[0].z;
                dx_o -= box_x * std::round(dx_o / box_x);
                dy_o -= box_y * std::round(dy_o / box_y);
                dz_o -= box_z * std::round(dz_o / box_z);
                double r_o = std::sqrt(dx_o * dx_o + dy_o * dy_o + dz_o * dz_o);

                double dx_n = new_pos.x - molecules[i].sites[0].x;
                double dy_n = new_pos.y - molecules[i].sites[0].y;
                double dz_n = new_pos.z - molecules[i].sites[0].z;
                dx_n -= box_x * std::round(dx_n / box_x);
                dy_n -= box_y * std::round(dy_n / box_y);
                dz_n -= box_z * std::round(dz_n / box_z);
                double r_n = std::sqrt(dx_n * dx_n + dy_n * dy_n + dz_n * dz_n);

                delta_E += pair_potentials[sp][molecules[i].species].evaluate(r_n) - pair_potentials[sp][molecules[i].species].evaluate(r_o);
                if (delta_E > 1e10) break;
            }
        }

        std::vector<ComplexDouble> delta_k;
        if (delta_E < 1e10 && electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            Molecule old_m, new_m;
            old_m.sites.push_back(old_pos); old_m.species = sp;
            new_m.sites.push_back(new_pos); new_m.species = sp;
            std::vector<ComplexDouble> del_k_old, del_k_new;
            calc_mol_delta_rho_k(old_m, -1.0, del_k_old);
            calc_mol_delta_rho_k(new_m, +1.0, del_k_new);
            delta_k.resize(ewald_params.k_vectors.size());
            for (size_t k = 0; k < delta_k.size(); ++k) {
                delta_k[k] = {del_k_old[k].re + del_k_new[k].re, del_k_old[k].im + del_k_new[k].im};
            }
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
        }

        if (delta_E < 1e10 && (delta_E <= 0.0 || rng.uniform() < std::exp(-beta * delta_E))) {
            molecules[idx].sites[0] = new_pos;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < rho_k.size(); ++k) {
                    rho_k[k].re += delta_k[k].re;
                    rho_k[k].im += delta_k[k].im;
                }
            }
        }
    }
}

void SimulationEngine::step_abc() {
    double r_move = rng.uniform();
    double volume = box_x * box_y * box_z;

    if (r_move < prob_insert) {
        if (static_cast<int>(molecules.size()) >= max_molecules) return;
        Vec3 com(rng.uniform_range(0, box_x), rng.uniform_range(0, box_y), rng.uniform_range(0, box_z));
        Quaternion q = Quaternion::random(rng.uniform(), rng.uniform(), rng.uniform());
        Vec3 axis = q.rotate(Vec3(0, 0, 1.0));

        Molecule new_mol;
        new_mol.sites.push_back(com - axis * bond_length); // Site A (-q)
        new_mol.sites.push_back(com);                     // Site B (0)
        new_mol.sites.push_back(com + axis * bond_length); // Site C (+q)
        new_mol.species = 0;

        double delta_E = external_potentials[0].evaluate(new_mol.sites[0]) +
                         external_potentials[1].evaluate(new_mol.sites[1]) +
                         external_potentials[2].evaluate(new_mol.sites[2]);
        if (delta_E > 1e10) return;

        for (const auto& other : molecules) {
            for (int s1 = 0; s1 < 3; ++s1) {
                for (int s2 = 0; s2 < 3; ++s2) {
                    double dx = new_mol.sites[s1].x - other.sites[s2].x;
                    double dy = new_mol.sites[s1].y - other.sites[s2].y;
                    double dz = new_mol.sites[s1].z - other.sites[s2].z;
                    dx -= box_x * std::round(dx / box_x);
                    dy -= box_y * std::round(dy / box_y);
                    dz -= box_z * std::round(dz / box_z);
                    delta_E += pair_potentials[s1][s2].evaluate(std::sqrt(dx * dx + dy * dy + dz * dz));
                    if (delta_E > 1e10) return;
                }
            }
        }

        std::vector<ComplexDouble> delta_k;
        if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            calc_mol_delta_rho_k(new_mol, +1.0, delta_k);
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            delta_E += calc_mol_self_energy(0, 3);
        }

        double log_p = -beta * (delta_E - mu1) + std::log(volume) - std::log(molecules.size() + 1);
        double prob = (log_p < 80.0) ? std::exp(log_p) : 0.0;
        if (rng.uniform() < prob) {
            molecules.push_back(new_mol);
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < rho_k.size(); ++k) {
                    rho_k[k].re += delta_k[k].re;
                    rho_k[k].im += delta_k[k].im;
                }
            }
        }
    } else if (r_move < prob_insert + prob_delete && !molecules.empty()) {
        int idx = rng.randint(0, static_cast<int>(molecules.size()) - 1);
        const auto& mol = molecules[idx];

        double delta_E = -external_potentials[0].evaluate(mol.sites[0]) -
                         external_potentials[1].evaluate(mol.sites[1]) -
                         external_potentials[2].evaluate(mol.sites[2]);

        for (size_t i = 0; i < molecules.size(); ++i) {
            if (static_cast<int>(i) == idx) continue;
            for (int s1 = 0; s1 < 3; ++s1) {
                for (int s2 = 0; s2 < 3; ++s2) {
                    double dx = mol.sites[s1].x - molecules[i].sites[s2].x;
                    double dy = mol.sites[s1].y - molecules[i].sites[s2].y;
                    double dz = mol.sites[s1].z - molecules[i].sites[s2].z;
                    dx -= box_x * std::round(dx / box_x);
                    dy -= box_y * std::round(dy / box_y);
                    dz -= box_z * std::round(dz / box_z);
                    delta_E -= pair_potentials[s1][s2].evaluate(std::sqrt(dx * dx + dy * dy + dz * dz));
                }
            }
        }

        std::vector<ComplexDouble> delta_k;
        if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            calc_mol_delta_rho_k(mol, -1.0, delta_k);
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
            delta_E -= calc_mol_self_energy(0, 3);
        }

        double log_p = -beta * (delta_E + mu1) + std::log(molecules.size()) - std::log(volume);
        double prob = (log_p < 80.0) ? std::exp(log_p) : 0.0;
        if (rng.uniform() < prob) {
            molecules[idx] = molecules.back();
            molecules.pop_back();
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < rho_k.size(); ++k) {
                    rho_k[k].re += delta_k[k].re;
                    rho_k[k].im += delta_k[k].im;
                }
            }
        }
    } else if (!molecules.empty()) {
        // Displace or rotate
        int idx = rng.randint(0, static_cast<int>(molecules.size()) - 1);
        Molecule old_mol = molecules[idx];
        Molecule new_mol = old_mol;

        if (rng.uniform() < 0.5) {
            // Displace
            Vec3 displ(rng.uniform_range(-maxdispl, maxdispl), rng.uniform_range(-maxdispl, maxdispl), rng.uniform_range(-maxdispl, maxdispl));
            for (int s = 0; s < 3; ++s) {
                new_mol.sites[s] += displ;
                new_mol.sites[s].x -= box_x * std::floor(new_mol.sites[s].x / box_x);
                new_mol.sites[s].y -= box_y * std::floor(new_mol.sites[s].y / box_y);
                new_mol.sites[s].z -= box_z * std::floor(new_mol.sites[s].z / box_z);
            }
        } else {
            // Rotate around center site B
            Vec3 center = old_mol.sites[1];
            Quaternion q = Quaternion::random(rng.uniform(), rng.uniform(), rng.uniform());
            for (int s = 0; s < 3; ++s) {
                Vec3 rel = old_mol.sites[s] - center;
                new_mol.sites[s] = center + q.rotate(rel);
                new_mol.sites[s].x -= box_x * std::floor(new_mol.sites[s].x / box_x);
                new_mol.sites[s].y -= box_y * std::floor(new_mol.sites[s].y / box_y);
                new_mol.sites[s].z -= box_z * std::floor(new_mol.sites[s].z / box_z);
            }
        }

        double delta_E = 0.0;
        for (int s = 0; s < 3; ++s) {
            delta_E += external_potentials[s].evaluate(new_mol.sites[s]) - external_potentials[s].evaluate(old_mol.sites[s]);
        }
        if (delta_E < 1e10) {
            for (size_t i = 0; i < molecules.size(); ++i) {
                if (static_cast<int>(i) == idx) continue;
                for (int s1 = 0; s1 < 3; ++s1) {
                    for (int s2 = 0; s2 < 3; ++s2) {
                        double dx_o = old_mol.sites[s1].x - molecules[i].sites[s2].x;
                        double dy_o = old_mol.sites[s1].y - molecules[i].sites[s2].y;
                        double dz_o = old_mol.sites[s1].z - molecules[i].sites[s2].z;
                        dx_o -= box_x * std::round(dx_o / box_x);
                        dy_o -= box_y * std::round(dy_o / box_y);
                        dz_o -= box_z * std::round(dz_o / box_z);
                        double r_o = std::sqrt(dx_o * dx_o + dy_o * dy_o + dz_o * dz_o);

                        double dx_n = new_mol.sites[s1].x - molecules[i].sites[s2].x;
                        double dy_n = new_mol.sites[s1].y - molecules[i].sites[s2].y;
                        double dz_n = new_mol.sites[s1].z - molecules[i].sites[s2].z;
                        dx_n -= box_x * std::round(dx_n / box_x);
                        dy_n -= box_y * std::round(dy_n / box_y);
                        dz_n -= box_z * std::round(dz_n / box_z);
                        double r_n = std::sqrt(dx_n * dx_n + dy_n * dy_n + dz_n * dz_n);

                        delta_E += pair_potentials[s1][s2].evaluate(r_n) - pair_potentials[s1][s2].evaluate(r_o);
                        if (delta_E > 1e10) break;
                    }
                }
            }
        }

        std::vector<ComplexDouble> delta_k;
        if (delta_E < 1e10 && electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
            std::vector<ComplexDouble> del_old, del_new;
            calc_mol_delta_rho_k(old_mol, -1.0, del_old);
            calc_mol_delta_rho_k(new_mol, +1.0, del_new);
            delta_k.resize(ewald_params.k_vectors.size());
            for (size_t k = 0; k < delta_k.size(); ++k) {
                delta_k[k] = {del_old[k].re + del_new[k].re, del_old[k].im + del_new[k].im};
            }
            delta_E += calc_ewald_reciprocal_energy_delta(delta_k);
        }

        if (delta_E < 1e10 && (delta_E <= 0.0 || rng.uniform() < std::exp(-beta * delta_E))) {
            molecules[idx] = new_mol;
            if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
                for (size_t k = 0; k < rho_k.size(); ++k) {
                    rho_k[k].re += delta_k[k].re;
                    rho_k[k].im += delta_k[k].im;
                }
            }
        }
    }
}

void SimulationEngine::step_h2o() {
    // Water step follows the rigid 3-site molecular move
    step_abc();
}

void SimulationEngine::step_co2() {
    // CO2 step follows the linear 3-site triatomic move
    step_abc();
}

double SimulationEngine::total_energy() const {
    double u_total = 0.0;

    // External energy
    for (const auto& mol : molecules) {
        int num_sites = static_cast<int>(mol.sites.size());
        for (int s = 0; s < num_sites; ++s) {
            u_total += external_potentials[s].evaluate(mol.sites[s]);
        }
    }

    // Pair energy
    for (size_t i = 0; i < molecules.size(); ++i) {
        for (size_t j = i + 1; j < molecules.size(); ++j) {
            int n1 = static_cast<int>(molecules[i].sites.size());
            int n2 = static_cast<int>(molecules[j].sites.size());
            for (int s1 = 0; s1 < n1; ++s1) {
                for (int s2 = 0; s2 < n2; ++s2) {
                    double dx = molecules[i].sites[s1].x - molecules[j].sites[s2].x;
                    double dy = molecules[i].sites[s1].y - molecules[j].sites[s2].y;
                    double dz = molecules[i].sites[s1].z - molecules[j].sites[s2].z;
                    dx -= box_x * std::round(dx / box_x);
                    dy -= box_y * std::round(dy / box_y);
                    dz -= box_z * std::round(dz / box_z);
                    int t1 = (mol_type == MoleculeType::TWO_TYPE) ? molecules[i].species : s1;
                    int t2 = (mol_type == MoleculeType::TWO_TYPE) ? molecules[j].species : s2;
                    u_total += pair_potentials[t1][t2].evaluate(std::sqrt(dx * dx + dy * dy + dz * dz));
                }
            }
        }
    }

    // Ewald reciprocal + self energy
    if (electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD && !ewald_params.k_vectors.empty()) {
        for (size_t k = 0; k < ewald_params.k_vectors.size(); ++k) {
            double w = ewald_params.k_vectors[k].weight;
            double re = rho_k[k].re;
            double im = rho_k[k].im;
            u_total += 0.5 * w * (re * re + im * im);
        }
        for (const auto& mol : molecules) {
            u_total += calc_mol_self_energy(mol.species, static_cast<int>(mol.sites.size()));
        }
    }

    return u_total;
}

} // namespace dens_city
