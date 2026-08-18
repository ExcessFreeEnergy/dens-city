#include "dens_city_env.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

#define M_PI_F 3.14159265358979323846f
static const float KB = 1.380649e-23f;
static const float COULOMB_PREFACTOR_F = 1.67101e-19f; // J * A / e^2

static inline float env_rng_uniform(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 40) * (1.0f / 16777216.0f);
}

void cdft_env_compute_restructuring_phi_r(CCdftEnv* env) {
    int N = CDFT_ENV_GRID_SIZE;
    float L_z = env->L_z;
    float dz = env->dz;
    float kappa_inv = env->kappa_inv; // e.g. 4.5 A

    for (int i = 0; i < N; ++i) {
        env->phi_R[i] = env->V_ext[i];
    }

    int kmax = 16;
    for (int m = 1; m <= kmax; ++m) {
        float k = 2.0f * M_PI_F * (float)m / L_z;
        float k_sq = k * k;

        float n_k_re = 0.0f;
        float n_k_im = 0.0f;
        for (int i = 0; i < N; ++i) {
            float z = env->z_coords[i];
            float angle = k * z;
            n_k_re += env->n_charge[i] * cosf(angle) * dz;
            n_k_im += env->n_charge[i] * sinf(angle) * dz;
        }

        float screening = expf(-k_sq * kappa_inv * kappa_inv / 4.0f);
        float kernel = (4.0f * M_PI_F / k_sq) * screening * (1.0f / L_z) * COULOMB_PREFACTOR_F;

        for (int i = 0; i < N; ++i) {
            float z = env->z_coords[i];
            float angle = k * z;
            float phi_k = 2.0f * kernel * (n_k_re * cosf(angle) + n_k_im * sinf(angle));
            env->phi_R[i] += phi_k;
        }
    }
}

void cdft_env_picard_relaxation_step(CCdftEnv* env, float alpha_mix) {
    int N = CDFT_ENV_GRID_SIZE;
    float beta = env->beta;
    float rho_b = env->rho_bulk;

    float total_mass = 0.0f;
    float res_sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        float v_eff = env->phi_R[i];
        float c1 = env->c1_pred[i];

        // Euler-Lagrange relaxation in dimensionless potential units:
        // rho_target = rho_b * exp[-beta * V_eff + c1]
        float arg = -beta * v_eff + c1;
        if (arg > 5.0f) arg = 5.0f;
        if (arg < -10.0f) arg = -10.0f;

        float rho_target = rho_b * expf(arg);
        float diff = rho_target - env->rho[i];
        res_sum += diff * diff;

        env->rho[i] = (1.0f - alpha_mix) * env->rho[i] + alpha_mix * rho_target;
        total_mass += env->rho[i] * env->dz;

        if (i > 0 && i < N - 1) {
            float dphi_dz = (env->phi_R[i + 1] - env->phi_R[i - 1]) / (2.0f * env->dz);
            env->n_charge[i] = -1e-4f * env->rho[i] * dphi_dz;
        }
    }

    env->current_filling = total_mass / (env->L_z * rho_b);
    env->el_residual = sqrtf(res_sum / (float)N);
}

void cdft_env_reset(CCdftEnv* env, int env_idx) {
    (void)env_idx;
    uint64_t* rng = &env->rng_state;

    env->L_z = 20.0f + env_rng_uniform(rng) * 20.0f; // 20 - 40 Angstroms
    env->dz = env->L_z / (float)CDFT_ENV_GRID_SIZE;
    env->T = 280.0f + env_rng_uniform(rng) * 150.0f; // 280 - 430 K
    env->beta = 1.0f / (KB * env->T);
    env->rho_bulk = 0.033f; // ~0.033 molecules / A^3 for water
    env->kappa_inv = 4.5f;

    // Randomize reservoir chemical potential
    env->mu_target = (-4000.0f + env_rng_uniform(rng) * 2500.0f) * KB; // in Joules
    env->target_filling = 0.2f + env_rng_uniform(rng) * 0.6f; // 0.2 - 0.8

    env->phi_0 = 0.0f;
    env->mode_m = 1.0f;
    env->v_bias = 0.0f;

    for (int i = 0; i < CDFT_ENV_GRID_SIZE; ++i) {
        env->z_coords[i] = (float)i * env->dz;
        env->rho[i] = env->rho_bulk * env->target_filling;
        env->n_charge[i] = 0.0f;
        env->V_ext[i] = 0.0f;
        env->phi_R[i] = 0.0f;
        env->c1_pred[i] = 0.0f;
    }

    env->current_filling = env->target_filling;
    env->el_residual = 0.0f;
    env->reward = 0.0f;
    env->done = false;
    env->step_count = 0;
    env->max_steps = 100;
}

void cdft_env_step(CCdftEnv* env, int env_idx) {
    (void)env_idx;
    int N = CDFT_ENV_GRID_SIZE;
    float L_z = env->L_z;

    // Update external potential from action parameters:
    // V_ext(z) = e * [ (phi_0 / m) * cos(2*pi*m*z / L_z) + v_bias * (z / L_z - 0.5) ]
    for (int i = 0; i < N; ++i) {
        float z = env->z_coords[i];
        float m = (env->mode_m > 0.5f) ? env->mode_m : 1.0f;
        float v_harmonic = (env->phi_0 / m) * cosf(2.0f * M_PI_F * m * z / L_z);
        float v_dc = env->v_bias * (z / L_z - 0.5f);
        env->V_ext[i] = (v_harmonic + v_dc) * 1e-21f; // Scaled to ~k_B*T order in Joules
    }

    // 1. Calculate 1D Fourier restructuring potential
    cdft_env_compute_restructuring_phi_r(env);

    // 2. Perform self-consistent Picard relaxation
    cdft_env_picard_relaxation_step(env, 0.15f);

    // 3. Compute continuous reward:
    // Balance between tracking target filling fraction theta* and minimizing EL residual
    float filling_err = fabsf(env->current_filling - env->target_filling);
    float r_filling = -10.0f * (filling_err * filling_err);
    float r_stability = -1.0f * env->el_residual;
    float r_action_cost = -0.01f * (env->phi_0 * env->phi_0 + env->v_bias * env->v_bias);

    env->reward = r_filling + r_stability + r_action_cost;
    env->step_count++;
    env->done = (env->step_count >= env->max_steps);
}

CCdftEnv* cdft_env_create(int num_envs, uint64_t seed) {
    CCdftEnv* envs = (CCdftEnv*)calloc((size_t)num_envs, sizeof(CCdftEnv));
    if (!envs) return NULL;

    for (int i = 0; i < num_envs; ++i) {
        envs[i].rng_state = seed + (uint64_t)i * 0x9e3779b97f4a7c15ULL;
        cdft_env_reset(&envs[i], i);
    }
    return envs;
}

void cdft_env_destroy(CCdftEnv* envs) {
    if (envs) {
        free(envs);
    }
}
