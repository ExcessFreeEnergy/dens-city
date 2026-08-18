#ifndef DENS_CITY_ENV_H
#define DENS_CITY_ENV_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDFT_ENV_GRID_SIZE 256
#define CDFT_MAX_K 128

typedef struct CCdftEnv {
    // 1. Thermodynamic & Slit Geometry Parameters
    float L_z;                   // Slit pore width in Angstroms (e.g. 20.0 to 100.0)
    float dz;                    // Grid spacing (L_z / CDFT_ENV_GRID_SIZE)
    float T;                     // Temperature in K
    float beta;                  // 1 / (kB * T) in J^-1
    float mu_target;             // Chemical potential mu in J
    float rho_bulk;              // Bulk reference density in A^-3
    float kappa_inv;             // Screening length in Angstroms (4.5 for water, 5.0 for RPM)

    // 2. Control Action Parameters
    float phi_0;                 // Applied voltage amplitude in Volts
    float mode_m;                // Harmonic spatial mode index m
    float v_bias;                // DC gate bias offset in Volts
    float target_filling;        // Desired target pore filling fraction theta*

    // 3. 1D State Profiles (Continuous Grid)
    float z_coords[CDFT_ENV_GRID_SIZE];
    float rho[CDFT_ENV_GRID_SIZE];       // Fluid density profile rho(z)
    float n_charge[CDFT_ENV_GRID_SIZE];  // Charge density profile n(z)
    float V_ext[CDFT_ENV_GRID_SIZE];     // Applied external potential V_ext(z)
    float phi_R[CDFT_ENV_GRID_SIZE];     // Embedded 1D Fourier restructuring potential
    float c1_pred[CDFT_ENV_GRID_SIZE];   // Predicted one-body direct correlation c^(1)(z)

    // 4. Dynamics & Reward Metrics
    float current_filling;       // Integral of rho(z) / (L_z * rho_bulk)
    float el_residual;           // Euler-Lagrange residual norm
    float reward;
    bool done;
    int step_count;
    int max_steps;

    // 5. Zero-Copy PufferLib Pointers
    float* observations;         // [rho(256), V_ext(256), phi_R(256), [T, mu, theta*](3)] -> 771 floats
    float* actions;              // [phi_0, mode_m, v_bias, c1_pred(256)]
    float* rewards;
    uint8_t* terminals;

    // 6. Fast RNG State
    uint64_t rng_state;
} CCdftEnv;

CCdftEnv* cdft_env_create(int num_envs, uint64_t seed);
void cdft_env_destroy(CCdftEnv* envs);
void cdft_env_reset(CCdftEnv* env, int env_idx);
void cdft_env_step(CCdftEnv* env, int env_idx);
void cdft_env_compute_restructuring_phi_r(CCdftEnv* env);
void cdft_env_picard_relaxation_step(CCdftEnv* env, float alpha_mix);

#ifdef __cplusplus
}
#endif

#endif // DENS_CITY_ENV_H
