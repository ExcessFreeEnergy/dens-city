#ifndef DENS_CITY_C_API_H
#define DENS_CITY_C_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* DensCityEngineHandle;

DensCityEngineHandle dens_city_create();
void dens_city_destroy(DensCityEngineHandle handle);

void dens_city_set_thermodynamics(DensCityEngineHandle handle, double T, double mu1, double mu2);
void dens_city_set_box(DensCityEngineHandle handle, double lx, double ly, double lz);
void dens_city_set_moves(DensCityEngineHandle handle, double p_ins, double p_del, double p_disp, double p_rot, double p_mut, double maxdispl, double maxrot);
void dens_city_set_molecule_type(DensCityEngineHandle handle, int mol_type, double bond_length);
void dens_city_set_electrostatics(DensCityEngineHandle handle, int mode, double ewald_alpha, int ewald_kmax);

void dens_city_set_pair_potential(
    DensCityEngineHandle handle,
    int type_i, int type_j,
    int kind,
    double epsilon_lj, double sigma_lj, double rc,
    double epsilon_c, double q1, double q2, double kappa_inv,
    double diameter, double prefactor, double shift_lj
);

void dens_city_set_pair_potential_buckingham(
    DensCityEngineHandle handle,
    int type_i, int type_j,
    double A_ij, double B_ij, double C_ij, double rc,
    double q1, double q2, double sigma_gauss_sq, double prefactor
);

void dens_city_set_external_potential(
    DensCityEngineHandle handle,
    int type_i,
    int kind,
    double low, double high, double width, double L,
    double epsilon, double sigma, double cutoff, double shift, double q,
    double A1, double A2, double A3, double A4,
    double phi1, double phi2, double phi3, double phi4
);

void dens_city_step(DensCityEngineHandle handle);
void dens_city_run_steps(DensCityEngineHandle handle, int n_steps);

int dens_city_get_molecule_count(DensCityEngineHandle handle);
double dens_city_get_total_energy(DensCityEngineHandle handle);

int dens_city_get_positions(DensCityEngineHandle handle, double* out_xyz, int* out_species, int max_mols);
int dens_city_add_molecule(DensCityEngineHandle handle, int species, const double* xyz_sites, int num_sites);

#ifdef __cplusplus
}
#endif

#endif // DENS_CITY_C_API_H
