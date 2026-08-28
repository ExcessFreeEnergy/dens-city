#ifndef CDFT_SOLVER_H
#define CDFT_SOLVER_H

#include "mol_graph.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define N_GRID 128
#define MAX_CDFT_ITERS 150
#define CDFT_TOLERANCE 1e-3f
#define CDFT_MIX_ALPHA 0.10f

// Physical parameters of the fluid
typedef struct {
    float sigma_eff;
    float epsilon_k_eff;
    float temperature_k;
    float beta;           // 1.0 / (k_B * T) in units of 1 / Kelvin
    float bulk_density;   // molecules / Å^3
    float slit_width_a;
    float dz;
    float mu_ex;          // Excess chemical potential in k_B * T
    float p_bulk_bar;     // Bulk reservoir pressure in bar
} FluidCDFTParams;

#define MAX_K_HALF 48

// Precomputed, 64-byte cache-aligned memory blocks for single environment execution (zero malloc in loop)
typedef struct {
    // 1D FMT Planar Weight Kernels (Analytical cell-integrated)
    float w0[N_GRID]   __attribute__((aligned(64)));
    float w1[N_GRID]   __attribute__((aligned(64)));
    float w2[N_GRID]   __attribute__((aligned(64)));
    float w3[N_GRID]   __attribute__((aligned(64)));
    float wv1[N_GRID]  __attribute__((aligned(64)));
    float wv2[N_GRID]  __attribute__((aligned(64)));

    // Symmetric compact support kernels for bounded zero-padded SIMD convolutions
    int k_half_fmt;
    int k_half_att;
    float w0_sym[2 * MAX_K_HALF + 1]   __attribute__((aligned(64)));
    float w1_sym[2 * MAX_K_HALF + 1]   __attribute__((aligned(64)));
    float w2_sym[2 * MAX_K_HALF + 1]   __attribute__((aligned(64)));
    float w3_sym[2 * MAX_K_HALF + 1]   __attribute__((aligned(64)));
    float wv1_sym[2 * MAX_K_HALF + 1]  __attribute__((aligned(64)));
    float wv2_sym[2 * MAX_K_HALF + 1]  __attribute__((aligned(64)));
    float wca_sym[2 * MAX_K_HALF + 1]  __attribute__((aligned(64)));

    // WCA Attractive Dispersion Kernel & Slit Wall External Potential
    float wca_kernel[N_GRID] __attribute__((aligned(64)));
    float v_ext[N_GRID]      __attribute__((aligned(64)));
    float dv_ext_dz[N_GRID]  __attribute__((aligned(64)));

    // Execution Buffers
    float rho[N_GRID]     __attribute__((aligned(64)));
    float rho_new[N_GRID] __attribute__((aligned(64)));
    float v_eff[N_GRID]   __attribute__((aligned(64)));
    float v_fmt[N_GRID]   __attribute__((aligned(64)));
    float v_wca[N_GRID]   __attribute__((aligned(64)));
    float pad_buf[N_GRID + 2 * MAX_K_HALF] __attribute__((aligned(64)));

    // FMT Weighted Densities n[6][N_GRID] and functional derivatives df_dn[6][N_GRID]
    // 0: n0, 1: n1, 2: n2, 3: n3, 4: nv1, 5: nv2
    float n[6][N_GRID]     __attribute__((aligned(64)));
    float df_dn[6][N_GRID] __attribute__((aligned(64)));
} CDFT_Env_Context;

typedef struct {
    float p_wall_bar;
    float omega_solv_kcal;
    float bulk_density;
    float packing_fraction;
    float peak_density;
    int converged;
    int iterations_taken;
} CDFT_Result;

// -------------------------------------------------------------
// 3D -> 1D Molecular Parameter Reduction & Equation of State
// -------------------------------------------------------------
static inline void derive_fluid_parameters_from_graph(const MolecularGraph* graph, FluidCDFTParams* params) {
    int n_atoms = graph->num_atoms;
    if (n_atoms == 0) {
        params->sigma_eff = 3.40f;
        params->epsilon_k_eff = 120.0f;
    } else if (n_atoms == 1) {
        params->sigma_eff = graph->atoms[0].sigma;
        params->epsilon_k_eff = graph->atoms[0].epsilon_k;
    } else {
        // Volume-equivalent hard-sphere diameter: sigma_eff = ( sum sigma_i^3 )^(1/3)
        float sig3_sum = 0.0f;
        for (int i = 0; i < n_atoms; i++) {
            float s = graph->atoms[i].sigma;
            if (s > 0.0f) sig3_sum += (s * s * s);
        }
        params->sigma_eff = (sig3_sum > 0.0f) ? cbrtf(sig3_sum) : 3.40f;

        // Exact WCA dispersion matching: eps_eff * sigma_eff^3 = sum_i sum_j sqrt(eps_i * eps_j) * ((sig_i + sig_j)/2)^3
        float att_vol_sum = 0.0f;
        for (int i = 0; i < n_atoms; i++) {
            for (int j = 0; j < n_atoms; j++) {
                float eps_ij = sqrtf(fmaxf(0.0f, graph->atoms[i].epsilon_k * graph->atoms[j].epsilon_k));
                float sig_ij = 0.5f * (graph->atoms[i].sigma + graph->atoms[j].sigma);
                att_vol_sum += eps_ij * (sig_ij * sig_ij * sig_ij);
            }
        }
        float sig_eff3 = params->sigma_eff * params->sigma_eff * params->sigma_eff;
        params->epsilon_k_eff = (sig_eff3 > 0.0f) ? (att_vol_sum / sig_eff3) : 120.0f;
    }

    params->temperature_k = 300.0f;
    params->beta = 1.0f / params->temperature_k; // in units of 1 / (k_B * T)
    params->slit_width_a = fmaxf(40.0f, 12.0f * params->sigma_eff);
    params->dz = params->slit_width_a / (float)N_GRID;

    // Self-consistent Percus-Yevick / Carnahan-Starling Equation of State solver for bulk density rho_bulk at P = 1.0 bar
    float b_vol = ((float)M_PI / 6.0f) * (params->sigma_eff * params->sigma_eff * params->sigma_eff);
    // WCA 3D dispersion volume integral: \int v_att(r) d^3r = 16 pi eps sigma^3 * [ -2 sqrt(2)/9 - 1/(9 r_cut^9) + 1/(3 r_cut^3) ]
    float r_cut = 5.0f;
    float prefactor = 16.0f * (float)M_PI * params->epsilon_k_eff * (params->sigma_eff * params->sigma_eff * params->sigma_eff);
    float bracket = -(2.0f * sqrtf(2.0f)) / 9.0f - 1.0f / (9.0f * powf(r_cut, 9.0f)) + 1.0f / (3.0f * powf(r_cut, 3.0f));
    float v_att_int = prefactor * bracket; // in K * Å^3 (negative)
    float a_att = -v_att_int / params->temperature_k; // in Å^3 (positive)

    // Solve for liquid packing fraction eta in [0.15, 0.48]
    float p_target_kbt = (1.0f * 1e5f) / (1.380649e-23f * params->temperature_k * 1e30f); // 1 bar in k_B T / Å^3
    float low = 0.01f / b_vol;
    float high = 0.55f / b_vol;
    float rho_sol = 0.02f;

    for (int iter = 0; iter < 50; iter++) {
        float mid = 0.5f * (low + high);
        float eta = b_vol * mid;
        float one_minus_eta = fmaxf(1e-6f, 1.0f - eta);
        float z_py = (1.0f + eta + eta * eta) / (one_minus_eta * one_minus_eta * one_minus_eta);
        float p_mid = mid * z_py - 0.5f * a_att * (mid * mid);

        if (fabsf(p_mid - p_target_kbt) < 1e-9f || (high - low) < 1e-8f) {
            rho_sol = mid;
            break;
        }
        if (p_mid < p_target_kbt) low = mid;
        else high = mid;
    }
    params->bulk_density = rho_sol;
    params->p_bulk_bar = 1.0f;

    // Excess chemical potential: mu_ex = mu_fmt + mu_att
    float eta_b = b_vol * params->bulk_density;
    float one_minus_eta_b = fmaxf(1e-6f, 1.0f - eta_b);
    float mu_fmt = -logf(one_minus_eta_b) + (eta_b * (14.0f - 13.0f * eta_b + 5.0f * (eta_b * eta_b))) / (2.0f * (one_minus_eta_b * one_minus_eta_b * one_minus_eta_b));
    float mu_att = -a_att * params->bulk_density;
    params->mu_ex = mu_fmt + mu_att;
}

// -------------------------------------------------------------
// Precompute Analytical Planar Kernels & Slit Wall Potential
// -------------------------------------------------------------
static inline void init_cdft_context(CDFT_Env_Context* ctx, const FluidCDFTParams* params) {
    memset(ctx, 0, sizeof(CDFT_Env_Context));

    float sigma = params->sigma_eff;
    float eps_k = params->epsilon_k_eff;
    float dz = params->dz;
    float R = sigma * 0.5f;

    int k_half = (int)ceilf(R / dz) + 1;
    if (k_half > MAX_K_HALF) k_half = MAX_K_HALF;
    ctx->k_half_fmt = k_half;

    // 1. Analytical Cell-Integrated FMT Kernels
    for (int i = -k_half; i <= k_half; i++) {
        int idx = (i + N_GRID) % N_GRID;
        float z_center = (float)i * dz;
        float z_left = z_center - 0.5f * dz;
        float z_right = z_center + 0.5f * dz;

        float z1 = fmaxf(-R, z_left);
        float z2 = fminf(R, z_right);

        if (z1 < z2) {
            float int_w3 = (float)M_PI * (R * R * (z2 - z1) - (z2 * z2 * z2 - z1 * z1 * z1) / 3.0f);
            float int_w2 = 2.0f * (float)M_PI * R * (z2 - z1);
            float int_wv2 = (float)M_PI * (z2 * z2 - z1 * z1);

            ctx->w3[idx] = int_w3 / dz;
            ctx->w2[idx] = int_w2 / dz;
            ctx->wv2[idx] = int_wv2 / dz;

            ctx->w3_sym[i + k_half] = int_w3 / dz;
            ctx->w2_sym[i + k_half] = int_w2 / dz;
            ctx->wv2_sym[i + k_half] = int_wv2 / dz;
        }
    }

    float denom_4piR = 4.0f * (float)M_PI * R;
    float denom_4piR2 = 4.0f * (float)M_PI * R * R;
    for (int i = 0; i < N_GRID; i++) {
        ctx->w1[i] = ctx->w2[i] / denom_4piR;
        ctx->w0[i] = ctx->w2[i] / denom_4piR2;
        ctx->wv1[i] = ctx->wv2[i] / denom_4piR;
    }
    for (int k = 0; k <= 2 * k_half; k++) {
        ctx->w1_sym[k] = ctx->w2_sym[k] / denom_4piR;
        ctx->w0_sym[k] = ctx->w2_sym[k] / denom_4piR2;
        ctx->wv1_sym[k] = ctx->wv2_sym[k] / denom_4piR;
    }

    // 2. Analytical 1D WCA Attractive Dispersion Kernel
    float r_cut = 5.0f * sigma;
    float r_min = powf(2.0f, 1.0f / 6.0f) * sigma;
    int att_half = (int)ceilf(r_cut / dz) + 1;
    if (att_half > MAX_K_HALF) att_half = MAX_K_HALF;
    ctx->k_half_att = att_half;

    for (int i = -att_half; i <= att_half; i++) {
        int idx = (i + N_GRID) % N_GRID;
        float z = fabsf((float)i * dz);

        if (z <= r_cut) {
            float v_1d = 0.0f;
            if (z < r_min) {
                // v_att,1D(z) = v_att,1D(r_min) + pi * eps * (r_min^2 - z^2)
                float v_rmin = 8.0f * (float)M_PI * eps_k * (-(powf(sigma, 12.0f) / (10.0f * powf(r_min, 10.0f))) + (powf(sigma, 6.0f) / (4.0f * powf(r_min, 4.0f)))) + (float)M_PI * eps_k * (r_min * r_min);
                v_1d = v_rmin - (float)M_PI * eps_k * (z * z);
            } else {
                v_1d = 8.0f * (float)M_PI * eps_k * (-(powf(sigma, 12.0f) / (10.0f * powf(z, 10.0f))) + (powf(sigma, 6.0f) / (4.0f * powf(z, 4.0f))));
            }
            ctx->wca_kernel[idx] = v_1d * dz / params->temperature_k; // in k_B * T units
            ctx->wca_sym[i + att_half] = v_1d * dz / params->temperature_k;
        }
    }

    // 3. Confining Slit-Pore Wall Potential (Steele 9-3 Wall Potential at z=0 and z=L)
    float wall_sigma = 3.40f;
    float wall_eps_kbt = 50.0f / params->temperature_k;
    float L_z = params->slit_width_a;

    for (int i = 0; i < N_GRID; i++) {
        float z = (float)i * dz + 0.5f * dz;
        float z_wall_left = z;
        float z_wall_right = L_z - z;

        float v_left = 0.0f;
        float v_right = 0.0f;
        float dv_left = 0.0f;
        float dv_right = 0.0f;

        // Left Wall: 9-3 potential
        if (z_wall_left < 0.45f * wall_sigma) {
            v_left = 1e5f;
            dv_left = -1e6f;
        } else {
            float sig_z = wall_sigma / z_wall_left;
            float sig_z3 = sig_z * sig_z * sig_z;
            float sig_z9 = sig_z3 * sig_z3 * sig_z3;
            v_left = (2.0f / 15.0f) * (float)M_PI * wall_eps_kbt * (2.0f * sig_z9 - 5.0f * sig_z3);
            dv_left = -(2.0f / 15.0f) * (float)M_PI * wall_eps_kbt * (18.0f * sig_z9 - 15.0f * sig_z3) / z_wall_left;
        }

        // Right Wall: 9-3 potential
        if (z_wall_right < 0.45f * wall_sigma) {
            v_right = 1e5f;
            dv_right = 1e6f;
        } else {
            float sig_z = wall_sigma / z_wall_right;
            float sig_z3 = sig_z * sig_z * sig_z;
            float sig_z9 = sig_z3 * sig_z3 * sig_z3;
            v_right = (2.0f / 15.0f) * (float)M_PI * wall_eps_kbt * (2.0f * sig_z9 - 5.0f * sig_z3);
            dv_right = (2.0f / 15.0f) * (float)M_PI * wall_eps_kbt * (18.0f * sig_z9 - 15.0f * sig_z3) / z_wall_right;
        }

        ctx->v_ext[i] = v_left + v_right;
        ctx->dv_ext_dz[i] = dv_left + dv_right;
    }
}

// -------------------------------------------------------------
// Vectorized SIMD 1D Zero-Padded Linear Bounded Convolution
// (Slit hard wall boundary truncation without circular wrap-around)
// -------------------------------------------------------------
static inline void simd_conv1d_bounded(
    const float* restrict input,
    const float* restrict kernel_sym,
    int k_half,
    float* restrict output,
    float* restrict pad_buf
) {
    memset(pad_buf, 0, k_half * sizeof(float));
    memcpy(&pad_buf[k_half], input, N_GRID * sizeof(float));
    memset(&pad_buf[k_half + N_GRID], 0, k_half * sizeof(float));

    for (int i = 0; i < N_GRID; i++) {
        float sum = 0.0f;
        const float* in_ptr = &pad_buf[i];
        #pragma omp simd reduction(+:sum)
        for (int k = 0; k <= 2 * k_half; k++) {
            sum += in_ptr[2 * k_half - k] * kernel_sym[k];
        }
        output[i] = sum;
    }
}

// -------------------------------------------------------------
// Fast cDFT Picard Iteration Solver Step
// -------------------------------------------------------------
static inline CDFT_Result solve_cdft_pufferlib_step(CDFT_Env_Context* ctx, const FluidCDFTParams* params) {
    CDFT_Result result;
    memset(&result, 0, sizeof(CDFT_Result));

    // Initialize density guess: Boltzmann profile rho(z) = rho_bulk * exp(-V_ext(z))
    for (int i = 0; i < N_GRID; i++) {
        float psi_0 = fminf(1.0f, -ctx->v_ext[i]);
        ctx->rho[i] = params->bulk_density * expf(psi_0);
    }

    float max_diff = 1.0f;
    int iter = 0;
    int k_fmt = ctx->k_half_fmt;
    int k_att = ctx->k_half_att;

    while (max_diff > CDFT_TOLERANCE && iter < MAX_CDFT_ITERS) {
        max_diff = 0.0f;

        // A. Forward FMT Bounded Convolutions
        simd_conv1d_bounded(ctx->rho, ctx->w0_sym, k_fmt, ctx->n[0], ctx->pad_buf);
        simd_conv1d_bounded(ctx->rho, ctx->w1_sym, k_fmt, ctx->n[1], ctx->pad_buf);
        simd_conv1d_bounded(ctx->rho, ctx->w2_sym, k_fmt, ctx->n[2], ctx->pad_buf);
        simd_conv1d_bounded(ctx->rho, ctx->w3_sym, k_fmt, ctx->n[3], ctx->pad_buf);
        simd_conv1d_bounded(ctx->rho, ctx->wv1_sym, k_fmt, ctx->n[4], ctx->pad_buf);
        simd_conv1d_bounded(ctx->rho, ctx->wv2_sym, k_fmt, ctx->n[5], ctx->pad_buf);

        // B. Rosenfeld FMT Free Energy Density Derivatives (df/dn_alpha)
        #pragma omp simd
        for (int i = 0; i < N_GRID; i++) {
            float n0 = ctx->n[0][i];
            float n1 = ctx->n[1][i];
            float n2 = ctx->n[2][i];
            float n3 = fminf(0.9999f, fmaxf(0.0f, ctx->n[3][i]));
            float nv1 = ctx->n[4][i];
            float nv2 = ctx->n[5][i];

            float one_m_n3 = 1.0f - n3;
            float one_m_n3_2 = one_m_n3 * one_m_n3;
            float one_m_n3_3 = one_m_n3_2 * one_m_n3;

            // df/dn0 = -ln(1 - n3)
            ctx->df_dn[0][i] = -logf(one_m_n3);
            // df/dn1 = n2 / (1 - n3)
            ctx->df_dn[1][i] = n2 / one_m_n3;
            // df/dn2 = n1 / (1 - n3) + (3 n2^2 - 3 nv2^2) / (24 pi (1 - n3)^2)
            ctx->df_dn[2][i] = (n1 / one_m_n3) + (3.0f * n2 * n2 - 3.0f * nv2 * nv2) / (24.0f * (float)M_PI * one_m_n3_2);
            // df/dn3 = n0/(1-n3) + (n1 n2 - nv1 nv2)/(1-n3)^2 + (n2^3 - 3 n2 nv2^2)/(12 pi (1-n3)^3)
            ctx->df_dn[3][i] = (n0 / one_m_n3) + (n1 * n2 - nv1 * nv2) / one_m_n3_2 + (n2 * n2 * n2 - 3.0f * n2 * nv2 * nv2) / (12.0f * (float)M_PI * one_m_n3_3);
            // df/dnv1 = -nv2 / (1 - n3)
            ctx->df_dn[4][i] = -nv2 / one_m_n3;
            // df/dnv2 = -nv1 / (1 - n3) - (6 n2 nv2) / (24 pi (1 - n3)^2)
            ctx->df_dn[5][i] = (-nv1 / one_m_n3) - (n2 * nv2) / (4.0f * (float)M_PI * one_m_n3_2);
        }

        // C. Reverse Bounded Convolutions to Accumulate V_FMT
        memset(ctx->v_fmt, 0, sizeof(ctx->v_fmt));
        float conv_buf[N_GRID] __attribute__((aligned(64)));

        simd_conv1d_bounded(ctx->df_dn[0], ctx->w0_sym, k_fmt, conv_buf, ctx->pad_buf);
        for (int i = 0; i < N_GRID; i++) ctx->v_fmt[i] += conv_buf[i];

        simd_conv1d_bounded(ctx->df_dn[1], ctx->w1_sym, k_fmt, conv_buf, ctx->pad_buf);
        for (int i = 0; i < N_GRID; i++) ctx->v_fmt[i] += conv_buf[i];

        simd_conv1d_bounded(ctx->df_dn[2], ctx->w2_sym, k_fmt, conv_buf, ctx->pad_buf);
        for (int i = 0; i < N_GRID; i++) ctx->v_fmt[i] += conv_buf[i];

        simd_conv1d_bounded(ctx->df_dn[3], ctx->w3_sym, k_fmt, conv_buf, ctx->pad_buf);
        for (int i = 0; i < N_GRID; i++) ctx->v_fmt[i] += conv_buf[i];

        simd_conv1d_bounded(ctx->df_dn[4], ctx->wv1_sym, k_fmt, conv_buf, ctx->pad_buf);
        for (int i = 0; i < N_GRID; i++) ctx->v_fmt[i] -= conv_buf[i]; // Vector reverse parity

        simd_conv1d_bounded(ctx->df_dn[5], ctx->wv2_sym, k_fmt, conv_buf, ctx->pad_buf);
        for (int i = 0; i < N_GRID; i++) ctx->v_fmt[i] -= conv_buf[i];

        // D. WCA Attractive Dispersion Bounded Convolution
        simd_conv1d_bounded(ctx->rho, ctx->wca_sym, k_att, ctx->v_wca, ctx->pad_buf);

        // E. Total Effective Potential & Picard Mixing
        #pragma omp simd
        for (int i = 0; i < N_GRID; i++) {
            ctx->v_eff[i] = ctx->v_ext[i] + ctx->v_fmt[i] + ctx->v_wca[i] - params->mu_ex;
            float psi = fminf(2.0f, -ctx->v_eff[i]);
            ctx->rho_new[i] = params->bulk_density * expf(psi);

            float diff = fabsf(ctx->rho_new[i] - ctx->rho[i]);
            if (diff > max_diff) max_diff = diff;

            ctx->rho[i] = CDFT_MIX_ALPHA * ctx->rho_new[i] + (1.0f - CDFT_MIX_ALPHA) * ctx->rho[i];
        }

        iter++;
    }

    result.iterations_taken = iter;
    result.converged = (max_diff <= CDFT_TOLERANCE);
    result.bulk_density = params->bulk_density;
    result.packing_fraction = ((float)M_PI / 6.0f) * params->bulk_density * (params->sigma_eff * params->sigma_eff * params->sigma_eff);

    // F. Exact Irving-Kirkwood Mechanical Virial Wall Pressure: P_wall = - \int_0^{L/2} rho(z) (dV_ext/dz) dz
    float f_virial = 0.0f;
    int mid = N_GRID / 2;
    for (int i = 0; i < mid; i++) {
        f_virial -= ctx->rho[i] * ctx->dv_ext_dz[i] * params->dz;
    }
    // 1 k_B T / Å^3 in bar: f_virial * (1.380649e-23 * 300 * 1e30 / 1e5) = f_virial * 41.419 bar
    result.p_wall_bar = f_virial * (1.380649e-23f * params->temperature_k * 1e25f);

    // G. Grand Potential Integral & Solvation Free Energy: Omega[rho]
    float grand_omega = 0.0f;
    float peak_rho = 0.0f;
    for (int i = 0; i < N_GRID; i++) {
        if (ctx->rho[i] > peak_rho) peak_rho = ctx->rho[i];
        // Ideal + Ext + fmt + att - mu*rho
        float f_id = ctx->rho[i] * (logf(fmaxf(1e-8f, ctx->rho[i] / params->bulk_density)) - 1.0f);
        float f_ext = ctx->rho[i] * ctx->v_ext[i];
        float f_att = 0.5f * ctx->rho[i] * ctx->v_wca[i];
        grand_omega += (f_id + f_ext + f_att - params->mu_ex * ctx->rho[i]) * params->dz;
    }
    result.peak_density = peak_rho;
    // 1 k_B * T = 1.987204e-3 * T kcal/mol = 0.59616 kcal/mol at 300 K
    result.omega_solv_kcal = grand_omega * (1.987204e-3f * params->temperature_k);

    return result;
}

#endif // CDFT_SOLVER_H
