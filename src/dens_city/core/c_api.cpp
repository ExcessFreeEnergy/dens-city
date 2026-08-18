#include "c_api.h"
#include "engine.h"

using namespace dens_city;

extern "C" {

DensCityEngineHandle dens_city_create() {
    return static_cast<DensCityEngineHandle>(new SimulationEngine());
}

void dens_city_destroy(DensCityEngineHandle handle) {
    if (!handle) return;
    delete static_cast<SimulationEngine*>(handle);
}

void dens_city_set_thermodynamics(DensCityEngineHandle handle, double T, double mu1, double mu2) {
    if (!handle) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    eng->T = T;
    eng->beta = 1.0 / (eng->kB * T);
    eng->mu1 = mu1;
    eng->mu2 = mu2;
}

void dens_city_set_box(DensCityEngineHandle handle, double lx, double ly, double lz) {
    if (!handle) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    eng->box_x = lx;
    eng->box_y = ly;
    eng->box_z = lz;
}

void dens_city_set_moves(DensCityEngineHandle handle, double p_ins, double p_del, double p_disp, double p_rot, double p_mut, double maxdispl, double maxrot) {
    if (!handle) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    eng->prob_insert = p_ins;
    eng->prob_delete = p_del;
    eng->prob_displace = p_disp;
    eng->prob_rotate = p_rot;
    eng->prob_mutate = p_mut;
    eng->maxdispl = maxdispl;
    eng->maxrot = maxrot;
}

void dens_city_set_molecule_type(DensCityEngineHandle handle, int mol_type, double bond_length) {
    if (!handle) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    eng->mol_type = static_cast<MoleculeType>(mol_type);
    eng->bond_length = bond_length;
}

void dens_city_set_electrostatics(DensCityEngineHandle handle, int mode, double ewald_alpha, int ewald_kmax) {
    if (!handle) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    eng->electrostatics_mode = static_cast<ElectrostaticsMode>(mode);
    if (eng->electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
        eng->init_ewald(ewald_alpha, ewald_kmax);
    }
}

void dens_city_set_pair_potential(
    DensCityEngineHandle handle,
    int type_i, int type_j,
    int kind,
    double epsilon_lj, double sigma_lj, double rc,
    double epsilon_c, double q1, double q2, double kappa_inv,
    double diameter, double prefactor, double shift_lj
) {
    if (!handle || type_i < 0 || type_i >= 4 || type_j < 0 || type_j >= 4) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    PairPotential pot;
    pot.kind = static_cast<PairPotentialKind>(kind);
    pot.epsilon_lj = epsilon_lj;
    pot.sigma_lj = sigma_lj;
    pot.rc = rc;
    pot.epsilon_c = epsilon_c;
    pot.q1 = q1;
    pot.q2 = q2;
    pot.kappa_inv = kappa_inv;
    pot.diameter = diameter;
    pot.prefactor = prefactor;
    pot.shift_lj = shift_lj;

    eng->pair_potentials[type_i][type_j] = pot;
    eng->pair_potentials[type_j][type_i] = pot;
}

void dens_city_set_external_potential(
    DensCityEngineHandle handle,
    int type_i,
    int kind,
    double low, double high, double width, double L,
    double epsilon, double sigma, double cutoff, double shift, double q,
    double A1, double A2, double A3, double A4,
    double phi1, double phi2, double phi3, double phi4
) {
    if (!handle || type_i < 0 || type_i >= 4) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    ExternalPotential ext;
    ext.kind = static_cast<ExternalPotentialKind>(kind);
    ext.low = low;
    ext.high = high;
    ext.width = width;
    ext.L = L;
    ext.epsilon = epsilon;
    ext.sigma = sigma;
    ext.cutoff = cutoff;
    ext.shift = shift;
    ext.q = q;
    ext.A1 = A1; ext.A2 = A2; ext.A3 = A3; ext.A4 = A4;
    ext.phi1 = phi1; ext.phi2 = phi2; ext.phi3 = phi3; ext.phi4 = phi4;

    eng->external_potentials[type_i] = ext;
}

void dens_city_step(DensCityEngineHandle handle) {
    if (!handle) return;
    static_cast<SimulationEngine*>(handle)->step();
}

void dens_city_run_steps(DensCityEngineHandle handle, int n_steps) {
    if (!handle) return;
    auto* eng = static_cast<SimulationEngine*>(handle);
    for (int i = 0; i < n_steps; ++i) {
        eng->step();
    }
}

int dens_city_get_molecule_count(DensCityEngineHandle handle) {
    if (!handle) return 0;
    return static_cast<SimulationEngine*>(handle)->get_molecule_count();
}

double dens_city_get_total_energy(DensCityEngineHandle handle) {
    if (!handle) return 0.0;
    return static_cast<SimulationEngine*>(handle)->total_energy();
}

int dens_city_get_positions(DensCityEngineHandle handle, double* out_xyz, int* out_species, int max_mols) {
    if (!handle || !out_xyz) return 0;
    auto* eng = static_cast<SimulationEngine*>(handle);
    int count = std::min(static_cast<int>(eng->molecules.size()), max_mols);
    int site_idx = 0;
    for (int i = 0; i < count; ++i) {
        if (out_species) out_species[i] = eng->molecules[i].species;
        for (const auto& s : eng->molecules[i].sites) {
            out_xyz[site_idx * 3 + 0] = s.x;
            out_xyz[site_idx * 3 + 1] = s.y;
            out_xyz[site_idx * 3 + 2] = s.z;
            site_idx++;
        }
    }
    return count;
}

int dens_city_add_molecule(DensCityEngineHandle handle, int species, const double* xyz_sites, int num_sites) {
    if (!handle || !xyz_sites || num_sites <= 0) return -1;
    auto* eng = static_cast<SimulationEngine*>(handle);
    Molecule mol;
    mol.species = species;
    for (int s = 0; s < num_sites; ++s) {
        mol.sites.push_back(Vec3(xyz_sites[s * 3 + 0], xyz_sites[s * 3 + 1], xyz_sites[s * 3 + 2]));
    }
    eng->molecules.push_back(mol);
    if (eng->electrostatics_mode == ElectrostaticsMode::LONG_RANGE_EWALD) {
        eng->refresh_rho_k();
    }
    return static_cast<int>(eng->molecules.size()) - 1;
}

} // extern "C"
