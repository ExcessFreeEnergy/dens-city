#ifndef DENS_CITY_ENV_H
#define DENS_CITY_ENV_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CDFT_ENV_GRID_SIZE 256
#define CDFT_ANDERSON_DEPTH 4

typedef struct CCdftEnv {
    float L_z;
    float dz;
    float T;
    float beta;
    float mu_target;
    float rho_bulk;
    float kappa_inv;

    float phi_0;
    float mode_m;
    float v_bias;
    float target_filling;
    int curriculum_mode;

    float z_coords[CDFT_ENV_GRID_SIZE];
    float rho[CDFT_ENV_GRID_SIZE];
    float rho_true[CDFT_ENV_GRID_SIZE];
    float n_charge[CDFT_ENV_GRID_SIZE];
    float V_ext[CDFT_ENV_GRID_SIZE];
    float phi_R[CDFT_ENV_GRID_SIZE];
    float c1_pred[CDFT_ENV_GRID_SIZE];

    // Static stack/struct-allocated Anderson history buffers (ZERO heap allocations)
    float rho_hist[CDFT_ANDERSON_DEPTH][CDFT_ENV_GRID_SIZE];
    float res_hist[CDFT_ANDERSON_DEPTH][CDFT_ENV_GRID_SIZE];
    int hist_count;
    int hist_head;

    float current_filling;
    float el_residual;
    float reward;
    bool done;
    int step_count;
    int max_steps;

    float* observations;
    float* actions;
    float* rewards;
    uint8_t* terminals;
    uint64_t rng_state;
} CCdftEnv;

CCdftEnv* cdft_env_create(int num_envs, uint64_t seed);
void cdft_env_destroy(CCdftEnv* envs);
void cdft_env_reset(CCdftEnv* env, int env_idx);
void cdft_env_step(CCdftEnv* env, int env_idx);

#ifdef __cplusplus
}
#endif

#endif // DENS_CITY_ENV_H
