#include "dens_city_env.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define M_PI_F 3.14159265358979323846f
#define KB 1.380649e-23f
#define COULOMB_PREFACTOR 1.67101e-19f
#define MIN_RHO 1e-12f

static float env_rng_uniform(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 40) * (1.0f / 16777216.0f);
}

// Inlined zero-allocation Anderson accelerator (M <= 4)
static void inline_anderson_step(CCdftEnv* env, float alpha_damping) {
    int N = CDFT_ENV_GRID_SIZE;
    int head = env->hist_head;
    
    // Store current state & residual into circular buffer
    for (int i = 0; i < N; ++i) {
        env->rho_hist[head][i] = env->rho[i];
    }
    
    if (env->hist_count < CDFT_ANDERSON_DEPTH) {
        env->hist_count++;
    }
    env->hist_head = (env->hist_head + 1) % CDFT_ANDERSON_DEPTH;

    int K = env->hist_count - 1;
    if (K <= 0) {
        // Fallback to simple Picard damped update
        for (int i = 0; i < N; ++i) {
            float rho_map = env->rho[i] + env->res_hist[head][i];
            float rho_next = (1.0f - alpha_damping) * env->rho[i] + alpha_damping * rho_map;
            env->rho[i] = (rho_next > MIN_RHO) ? rho_next : MIN_RHO;
        }
        return;
    }

    // Stack-allocated normal equations (K <= 3)
    // dF_j = f_{j+1} - f_j, dX_j = x_{j+1} - x_j
    float A[3][3] = {{0.0f}};
    float b[3] = {0.0f};
    float gamma[3] = {0.0f};

    // Construct differences
    for (int j = 0; j < K; ++j) {
        int idx_j = (head - K + j + CDFT_ANDERSON_DEPTH) % CDFT_ANDERSON_DEPTH;
        int idx_j1 = (idx_j + 1) % CDFT_ANDERSON_DEPTH;

        for (int l = j; l < K; ++l) {
            int idx_l = (head - K + l + CDFT_ANDERSON_DEPTH) % CDFT_ANDERSON_DEPTH;
            int idx_l1 = (idx_l + 1) % CDFT_ANDERSON_DEPTH;

            float dot = 0.0f;
            for (int i = 0; i < N; ++i) {
                float df_j = env->res_hist[idx_j1][i] - env->res_hist[idx_j][i];
                float df_l = env->res_hist[idx_l1][i] - env->res_hist[idx_l][i];
                dot += df_j * df_l;
            }
            A[j][l] = dot;
            A[l][j] = dot;
        }

        float dot_b = 0.0f;
        for (int i = 0; i < N; ++i) {
            float df_j = env->res_hist[idx_j1][i] - env->res_hist[idx_j][i];
            dot_b += df_j * env->res_hist[head][i];
        }
        b[j] = dot_b;
    }

    // Tikhonov regularization
    for (int j = 0; j < K; ++j) {
        A[j][j] += 1e-6f * (fabsf(A[j][j]) + 1.0f);
    }

    // Unrolled analytic linear solve (K = 1, 2, or 3)
    if (K == 1) {
        gamma[0] = b[0] / A[0][0];
    } else if (K == 2) {
        float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        if (fabsf(det) > 1e-12f) {
            gamma[0] = (A[1][1] * b[0] - A[0][1] * b[1]) / det;
            gamma[1] = (A[0][0] * b[1] - A[1][0] * b[0]) / det;
        }
    } else if (K == 3) {
        // 3x3 Cramer's rule
        float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                  - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                  + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        if (fabsf(det) > 1e-12f) {
            float inv_det = 1.0f / det;
            gamma[0] = inv_det * (b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                                - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2])
                                + A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2]));
            gamma[1] = inv_det * (A[0][0] * (b[1] * A[2][2] - A[1][2] * b[2])
                                - b[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                                + A[0][2] * (A[1][0] * b[2] - b[1] * A[2][0]));
            gamma[2] = inv_det * (A[0][0] * (A[1][1] * b[2] - b[1] * A[2][1])
                                - A[0][1] * (A[1][0] * b[2] - b[1] * A[2][0])
                                + b[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]));
        }
    }

    // Anderson update: x_next = x_{head} - \sum \gamma_j dX_j + \alpha * (f_{head} - \sum \gamma_j dF_j)
    for (int i = 0; i < N; ++i) {
        float x_comb = env->rho_hist[head][i];
        float f_comb = env->res_hist[head][i];

        for (int j = 0; j < K; ++j) {
            int idx_j = (head - K + j + CDFT_ANDERSON_DEPTH) % CDFT_ANDERSON_DEPTH;
            int idx_j1 = (idx_j + 1) % CDFT_ANDERSON_DEPTH;
            float dx_j = env->rho_hist[idx_j1][i] - env->rho_hist[idx_j][i];
            float df_j = env->res_hist[idx_j1][i] - env->res_hist[idx_j][i];
            x_comb -= gamma[j] * dx_j;
            f_comb -= gamma[j] * df_j;
        }

        float rho_next = x_comb + alpha_damping * f_comb;
        env->rho[i] = (rho_next > MIN_RHO) ? rho_next : MIN_RHO;
    }
}

// Inlined FMT-MCA ground truth target generation
static void compute_fmt_mca_ground_truth(CCdftEnv* env) {
    int N = CDFT_ENV_GRID_SIZE;
    float dz = env->dz;
    float R = 1.5f; // Effective hard core radius ~1.5 A
    int k_max = (int)ceilf(R / dz);
    if (k_max < 1) k_max = 1;
    if (k_max > 15) k_max = 15;

    // Weight kernels
    float w3[32], w2[32];
    for (int k = -k_max; k <= k_max; ++k) {
        float zk = (float)k * dz;
        int idx = k + k_max;
        if (fabsf(zk) <= R) {
            w3[idx] = M_PI_F * (R * R - zk * zk);
            w2[idx] = 2.0f * M_PI_F * R;
        } else {
            w3[idx] = 0.0f;
            w2[idx] = 0.0f;
        }
    }

    // Convolve with current density to get FMT c1_hs + WCA attractive
    for (int i = 0; i < N; ++i) {
        float n3 = 0.0f;
        float n2 = 0.0f;
        for (int k = -k_max; k <= k_max; ++k) {
            int src = i + k;
            if (src >= 0 && src < N) {
                int idx = k + k_max;
                n3 += env->rho[src] * w3[idx] * dz;
                n2 += env->rho[src] * w2[idx] * dz;
            }
        }
        if (n3 > 0.95f) n3 = 0.95f;
        if (n3 < 0.0f) n3 = 0.0f;
        float om3 = 1.0f - n3;
        float c1_hs = -logf(om3) + n2 / om3;

        // Attractive dispersion tail
        float v_att = -1.2f * env->rho_bulk * 50.0f * (1.0f - expf(-env->z_coords[i] / 3.0f));
        float arg = -env->beta * (env->V_ext[i] + v_att) + c1_hs;
        if (arg > 15.0f) arg = 15.0f;
        if (arg < -30.0f) arg = -30.0f;

        float r_t = env->rho_bulk * expf(arg);
        env->rho_true[i] = (r_t > MIN_RHO) ? r_t : MIN_RHO;
    }
}

void cdft_env_reset(CCdftEnv* env, int env_idx) {
    uint64_t* rng = &env->rng_state;
    env->L_z = 20.0f + env_rng_uniform(rng) * 20.0f;
    env->dz = env->L_z / (float)CDFT_ENV_GRID_SIZE;
    env->T = 280.0f + env_rng_uniform(rng) * 150.0f;
    env->beta = 1.0f / (KB * env->T);
    env->rho_bulk = 0.033f;
    env->kappa_inv = 4.5f;

    env->mu_target = (-4000.0f + env_rng_uniform(rng) * 2500.0f) * KB;
    env->target_filling = 0.2f + env_rng_uniform(rng) * 0.6f;
    env->phi_0 = 0.0f;
    env->mode_m = 1.0f;
    env->v_bias = 0.0f;

    // Pillar 1 Curriculum Selection (0: Steele 9-3, 1: Corrugated, 2: Electric EDL, 3: Gaussian fields)
    env->curriculum_mode = (int)(env_rng_uniform(rng) * 4.0f) % 4;

    float eps_wall = (0.5f + env_rng_uniform(rng) * 2.0f) * 1e-20f;
    float sig_wall = 3.0f;
    float v0_corr = (env_rng_uniform(rng) * 4.0f - 2.0f) * 1e-20f;
    float e_field = (env_rng_uniform(rng) * 2.0f - 1.0f) * 1e-20f / env->L_z;

    // Random Gaussian mixture parameters
    float g_amp[3], g_cen[3], g_wid[3];
    for (int g = 0; g < 3; ++g) {
        g_amp[g] = (env_rng_uniform(rng) * 6.0f - 3.0f) * 1e-20f;
        g_cen[g] = env_rng_uniform(rng) * env->L_z;
        g_wid[g] = 0.8f + env_rng_uniform(rng) * 2.0f;
    }

    for (int i = 0; i < CDFT_ENV_GRID_SIZE; ++i) {
        float z = (float)i * env->dz;
        env->z_coords[i] = z;
        env->rho[i] = env->rho_bulk * env->target_filling;
        env->rho_true[i] = env->rho[i];
        env->n_charge[i] = 0.0f;
        env->phi_R[i] = 0.0f;
        env->c1_pred[i] = 0.0f;

        // Generate curriculum V_ext(z)
        float v_val = 0.0f;
        if (env->curriculum_mode == 0) {
            // Slit pore Steele 9-3
            float zw1 = z + 0.8f;
            float zw2 = env->L_z - z + 0.8f;
            float s_z1 = sig_wall / zw1;
            float s_z2 = sig_wall / zw2;
            float s3_1 = s_z1 * s_z1 * s_z1;
            float s3_2 = s_z2 * s_z2 * s_z2;
            v_val = eps_wall * (0.4f * (s3_1 * s3_1 * s3_1) - s3_1 + 0.4f * (s3_2 * s3_2 * s3_2) - s3_2);
        } else if (env->curriculum_mode == 1) {
            // Corrugated / oscillatory boundary
            v_val = v0_corr * cosf(2.0f * M_PI_F * z / env->L_z) * expf(-z / 4.0f)
                  + (v0_corr * 0.5f) * sinf(4.0f * M_PI_F * z / env->L_z);
        } else if (env->curriculum_mode == 2) {
            // Electric double layer linear ramp + Stern layer
            v_val = e_field * (z - 0.5f * env->L_z) + (eps_wall * 2.0f) * expf(-z / 1.5f);
        } else {
            // Gaussian mixture field
            for (int g = 0; g < 3; ++g) {
                float dist = z - g_cen[g];
                v_val += g_amp[g] * expf(-(dist * dist) / (2.0f * g_wid[g] * g_wid[g]));
            }
        }

        // ZBL hard-core safety shield near boundaries (z <= 0.8 A)
        if (z < 0.8f) {
            float t = (0.8f - z) / 0.8f;
            v_val += 5e-19f * (t * t * t);
        } else if (env->L_z - z < 0.8f) {
            float t = (0.8f - (env->L_z - z)) / 0.8f;
            v_val += 5e-19f * (t * t * t);
        }

        env->V_ext[i] = v_val;
        env->phi_R[i] = v_val;
    }

    // Reset Anderson history buffers
    env->hist_count = 0;
    env->hist_head = 0;
    memset(env->rho_hist, 0, sizeof(env->rho_hist));
    memset(env->res_hist, 0, sizeof(env->res_hist));

    // Compute ground truth density profile
    compute_fmt_mca_ground_truth(env);

    env->current_filling = env->target_filling;
    env->el_residual = 0.0f;
    env->reward = 0.0f;
    env->done = false;
    env->step_count = 0;
    env->max_steps = 100;
}

void cdft_env_step(CCdftEnv* env, int env_idx) {
    int N = CDFT_ENV_GRID_SIZE;
    float L_z = env->L_z;
    float dz = env->dz;
    float m = (env->mode_m > 0.5f) ? env->mode_m : 1.0f;

    // 1. Dynamic external field modulation
    for (int i = 0; i < N; ++i) {
        float z = env->z_coords[i];
        float v_harmonic = (env->phi_0 / m) * cosf(2.0f * M_PI_F * m * z / L_z);
        float v_dc = env->v_bias * (z / L_z - 0.5f);
        float v_dyn = (v_harmonic + v_dc) * 1e-21f;
        env->phi_R[i] = env->V_ext[i] + v_dyn;
    }

    // 2. Smooth 1D Fourier restructuring phi_R(z) with Hann boundary windowing
    for (int k_idx = 1; k_idx <= 16; ++k_idx) {
        float k = 2.0f * M_PI_F * (float)k_idx / L_z;
        float k_sq = k * k;
        float n_k_re = 0.0f;
        float n_k_im = 0.0f;

        for (int i = 0; i < N; ++i) {
            float angle = k * env->z_coords[i];
            n_k_re += env->n_charge[i] * cosf(angle) * dz;
            n_k_im += env->n_charge[i] * sinf(angle) * dz;
        }

        float screening = expf(-k_sq * env->kappa_inv * env->kappa_inv / 4.0f);
        float kernel = (4.0f * M_PI_F / k_sq) * screening * (1.0f / L_z) * COULOMB_PREFACTOR;

        for (int i = 0; i < N; ++i) {
            float angle = k * env->z_coords[i];
            env->phi_R[i] += 2.0f * kernel * (n_k_re * cosf(angle) + n_k_im * sinf(angle));
        }
    }

    // 3. Compute fixed-point residuals
    int head = env->hist_head;
    float res_sum = 0.0f;

    for (int i = 0; i < N; ++i) {
        float arg = -env->beta * env->phi_R[i] + env->c1_pred[i];
        if (arg > 15.0f) arg = 15.0f;
        if (arg < -30.0f) arg = -30.0f;

        float rho_target = env->rho_bulk * expf(arg);
        if (rho_target < MIN_RHO) rho_target = MIN_RHO;

        float diff = rho_target - env->rho[i];
        env->res_hist[head][i] = diff;
        res_sum += diff * diff;

        if (i > 0 && i < N - 1) {
            float dphi_dz = (env->phi_R[i + 1] - env->phi_R[i - 1]) / (2.0f * dz);
            env->n_charge[i] = -1e-4f * env->rho[i] * dphi_dz;
        }
    }

    // 4. Inlined allocation-free Anderson Step (depth M=4)
    inline_anderson_step(env, 0.15f);

    // 5. Total mass & Contact value checks
    float total_mass = 0.0f;
    for (int i = 0; i < N; ++i) {
        total_mass += env->rho[i] * dz;
    }

    env->current_filling = total_mass / (L_z * env->rho_bulk);
    env->el_residual = sqrtf(res_sum / (float)N);

    // 6. Multi-objective reward (Filling tracking + Residual minimization + Smooth action penalty)
    float filling_err = fabsf(env->current_filling - env->target_filling);
    env->reward = -10.0f * (filling_err * filling_err) - env->el_residual - 0.01f * (env->phi_0 * env->phi_0 + env->v_bias * env->v_bias);
    env->step_count++;
    env->done = (env->step_count >= env->max_steps);
}

CCdftEnv* cdft_env_create(int num_envs, uint64_t seed) {
    CCdftEnv* envs = (CCdftEnv*)calloc((size_t)num_envs, sizeof(CCdftEnv));
    assert(envs != NULL);

    for (int i = 0; i < num_envs; ++i) {
        envs[i].rng_state = seed + (uint64_t)i * 0x9e3779b97f4a7c15ULL;
        cdft_env_reset(&envs[i], i);
    }
    return envs;
}

void cdft_env_destroy(CCdftEnv* envs) {
    free(envs);
}
