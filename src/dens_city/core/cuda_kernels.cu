#include "cuda_engine.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

__device__ static float curand_uniform_f(uint64_t* state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (float)((*state * 0x2545F4914F6CDD1DULL) >> 40) * (1.0f / 16777216.0f);
}

__device__ static float evaluate_pair_potential_dev(CUDAPairPotential pot, float r) {
    if (r >= pot.rc) return 0.0f;
    if (r <= 1e-6f) return 1e10f;

    if (pot.kind == 1) {
        float s_r = pot.sigma_lj / r;
        float s_r6 = s_r * s_r * s_r * s_r * s_r * s_r;
        return 4.0f * pot.epsilon_lj * (s_r6 * s_r6 - s_r6) - pot.shift_lj;
    }
    if (pot.kind == 3) {
        return (r < pot.diameter) ? 1e10f : 0.0f;
    }
    if (pot.kind == 4) {
        if (r < pot.diameter) return 1e10f;
        return (pot.prefactor * pot.q1 * pot.q2 / r) * erfc(r / pot.kappa_inv);
    }
    if (pot.kind == 5) {
        float u_lj = 0.0f;
        if (r < pot.rc) {
            float s_r = pot.sigma_lj / r;
            float s_r6 = s_r * s_r * s_r * s_r * s_r * s_r;
            u_lj = 4.0f * pot.epsilon_lj * (s_r6 * s_r6 - s_r6) - pot.shift_lj;
        }
        float u_c = 0.0f;
        if (fabsf(pot.q1) > 1e-5f && fabsf(pot.q2) > 1e-5f) {
            u_c = (pot.prefactor * pot.q1 * pot.q2 / r) * erfc(r / pot.kappa_inv);
        }
        return u_lj + u_c;
    }
    return 0.0f;
}

__device__ static float evaluate_ext_potential_dev(CUDAExternalPotential ext, CUDAVec3 pos) {
    if (ext.kind == 1) {
        if (pos.z < ext.low || pos.z > ext.high) return 1e10f;
        return 0.0f;
    }
    if (ext.kind == 3) {
        if (pos.z < ext.low || pos.z > ext.high) return 1e10f;
        float u = 0.0f;
        if (fabsf(ext.q) > 1e-5f && ext.L > 0.0f) {
            if (fabsf(ext.A1) > 1e-5f) u += ext.q * ext.A1 * cosf(2.0f * 3.14159265f * 1.0f * pos.z / ext.L + ext.phi1);
            if (fabsf(ext.A2) > 1e-5f) u += ext.q * ext.A2 * cosf(2.0f * 3.14159265f * 2.0f * pos.z / ext.L + ext.phi2);
            if (fabsf(ext.A3) > 1e-5f) u += ext.q * ext.A3 * cosf(2.0f * 3.14159265f * 3.0f * pos.z / ext.L + ext.phi3);
            if (fabsf(ext.A4) > 1e-5f) u += ext.q * ext.A4 * cosf(2.0f * 3.14159265f * 4.0f * pos.z / ext.L + ext.phi4);
        }
        return u;
    }
    return 0.0f;
}

__global__ void batch_gcmc_kernel(
    int num_boxes,
    int steps_per_launch,
    CUDABoxConfig* configs,
    CUDAVec3* positions,
    int* species,
    int* mol_counts,
    uint64_t* rng_states,
    float* rho_k_re,
    float* rho_k_im,
    float* energies
) {
    int box_idx = blockIdx.x;
    if (box_idx >= num_boxes) return;

    __shared__ CUDABoxConfig s_cfg;
    if (threadIdx.x == 0) {
        s_cfg = configs[box_idx];
    }
    __syncthreads();

    uint64_t local_rng = rng_states[box_idx * 32 + threadIdx.x];
    int count = mol_counts[box_idx];
    CUDAVec3* box_pos = &positions[box_idx * CUDA_MAX_MOLS * CUDA_MAX_SITES];
    int* box_spec = &species[box_idx * CUDA_MAX_MOLS];
    float vol = s_cfg.box_x * s_cfg.box_y * s_cfg.box_z;

    for (int step = 0; step < steps_per_launch; ++step) {
        if (threadIdx.x == 0) {
            float r_move = curand_uniform_f(&local_rng);
            if (r_move < 0.33f) {
                // Insertion
                if (count < s_cfg.max_molecules) {
                    CUDAVec3 pos = {
                        curand_uniform_f(&local_rng) * s_cfg.box_x,
                        curand_uniform_f(&local_rng) * s_cfg.box_y,
                        curand_uniform_f(&local_rng) * s_cfg.box_z
                    };
                    float delta_E = evaluate_ext_potential_dev(s_cfg.external_potentials[0], pos);
                    if (delta_E < 1e9f) {
                        for (int i = 0; i < count; ++i) {
                            float dx = pos.x - box_pos[i].x;
                            float dy = pos.y - box_pos[i].y;
                            float dz = pos.z - box_pos[i].z;
                            dx -= s_cfg.box_x * roundf(dx / s_cfg.box_x);
                            dy -= s_cfg.box_y * roundf(dy / s_cfg.box_y);
                            dz -= s_cfg.box_z * roundf(dz / s_cfg.box_z);
                            float r = sqrtf(dx * dx + dy * dy + dz * dz);
                            delta_E += evaluate_pair_potential_dev(s_cfg.pair_potentials[0][0], r);
                            if (delta_E > 1e9f) break;
                        }
                    }
                    if (delta_E < 1e9f) {
                        float log_p = -s_cfg.beta * (delta_E - s_cfg.mu1) + logf(vol) - logf(count + 1.0f);
                        if (log_p > 0.0f || curand_uniform_f(&local_rng) < expf(log_p)) {
                            box_pos[count] = pos;
                            box_spec[count] = 0;
                            count++;
                        }
                    }
                }
            } else if (r_move < 0.66f && count > 0) {
                // Deletion
                int idx = (int)(curand_uniform_f(&local_rng) * count) % count;
                CUDAVec3 pos = box_pos[idx];
                float delta_E = -evaluate_ext_potential_dev(s_cfg.external_potentials[0], pos);
                for (int i = 0; i < count; ++i) {
                    if (i == idx) continue;
                    float dx = pos.x - box_pos[i].x;
                    float dy = pos.y - box_pos[i].y;
                    float dz = pos.z - box_pos[i].z;
                    dx -= s_cfg.box_x * roundf(dx / s_cfg.box_x);
                    dy -= s_cfg.box_y * roundf(dy / s_cfg.box_y);
                    dz -= s_cfg.box_z * roundf(dz / s_cfg.box_z);
                    float r = sqrtf(dx * dx + dy * dy + dz * dz);
                    delta_E -= evaluate_pair_potential_dev(s_cfg.pair_potentials[0][0], r);
                }
                float log_p = -s_cfg.beta * (delta_E + s_cfg.mu1) + logf((float)count) - logf(vol);
                if (log_p > 0.0f || curand_uniform_f(&local_rng) < expf(log_p)) {
                    box_pos[idx] = box_pos[count - 1];
                    box_spec[idx] = box_spec[count - 1];
                    count--;
                }
            } else if (count > 0) {
                // Displacement
                int idx = (int)(curand_uniform_f(&local_rng) * count) % count;
                CUDAVec3 old_pos = box_pos[idx];
                CUDAVec3 new_pos = {
                    old_pos.x + (curand_uniform_f(&local_rng) - 0.5f) * 2.0f * s_cfg.maxdispl,
                    old_pos.y + (curand_uniform_f(&local_rng) - 0.5f) * 2.0f * s_cfg.maxdispl,
                    old_pos.z + (curand_uniform_f(&local_rng) - 0.5f) * 2.0f * s_cfg.maxdispl
                };
                new_pos.x -= s_cfg.box_x * floorf(new_pos.x / s_cfg.box_x);
                new_pos.y -= s_cfg.box_y * floorf(new_pos.y / s_cfg.box_y);
                new_pos.z -= s_cfg.box_z * floorf(new_pos.z / s_cfg.box_z);

                float delta_E = evaluate_ext_potential_dev(s_cfg.external_potentials[0], new_pos) -
                                evaluate_ext_potential_dev(s_cfg.external_potentials[0], old_pos);
                if (delta_E < 1e9f) {
                    for (int i = 0; i < count; ++i) {
                        if (i == idx) continue;
                        float dx_o = old_pos.x - box_pos[i].x;
                        float dy_o = old_pos.y - box_pos[i].y;
                        float dz_o = old_pos.z - box_pos[i].z;
                        dx_o -= s_cfg.box_x * roundf(dx_o / s_cfg.box_x);
                        dy_o -= s_cfg.box_y * roundf(dy_o / s_cfg.box_y);
                        dz_o -= s_cfg.box_z * roundf(dz_o / s_cfg.box_z);
                        float r_o = sqrtf(dx_o * dx_o + dy_o * dy_o + dz_o * dz_o);

                        float dx_n = new_pos.x - box_pos[i].x;
                        float dy_n = new_pos.y - box_pos[i].y;
                        float dz_n = new_pos.z - box_pos[i].z;
                        dx_n -= s_cfg.box_x * roundf(dx_n / s_cfg.box_x);
                        dy_n -= s_cfg.box_y * roundf(dy_n / s_cfg.box_y);
                        dz_n -= s_cfg.box_z * roundf(dz_n / s_cfg.box_z);
                        float r_n = sqrtf(dx_n * dx_n + dy_n * dy_n + dz_n * dz_n);

                        delta_E += evaluate_pair_potential_dev(s_cfg.pair_potentials[0][0], r_n) -
                                   evaluate_pair_potential_dev(s_cfg.pair_potentials[0][0], r_o);
                        if (delta_E > 1e9f) break;
                    }
                }
                if (delta_E < 1e9f && (delta_E <= 0.0f || curand_uniform_f(&local_rng) < expf(-s_cfg.beta * delta_E))) {
                    box_pos[idx] = new_pos;
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        mol_counts[box_idx] = count;
        rng_states[box_idx * 32] = local_rng;
    }
}

extern "C" {

CUDABatchEngine* cuda_batch_create(int num_boxes, const CUDABoxConfig* configs, uint64_t seed) {
    CUDABatchEngine* engine = (CUDABatchEngine*)calloc(1, sizeof(CUDABatchEngine));
    assert(engine != NULL);
    engine->num_boxes = num_boxes;

    cudaMalloc(&engine->d_configs, num_boxes * sizeof(CUDABoxConfig));
    cudaMemcpy(engine->d_configs, configs, num_boxes * sizeof(CUDABoxConfig), cudaMemcpyHostToDevice);

    size_t pos_size = num_boxes * CUDA_MAX_MOLS * CUDA_MAX_SITES * sizeof(CUDAVec3);
    cudaMalloc(&engine->d_positions, pos_size);
    cudaMemset(engine->d_positions, 0, pos_size);

    cudaMalloc(&engine->d_species, num_boxes * CUDA_MAX_MOLS * sizeof(int));
    cudaMemset(engine->d_species, 0, num_boxes * CUDA_MAX_MOLS * sizeof(int));

    cudaMalloc(&engine->d_mol_counts, num_boxes * sizeof(int));
    cudaMemset(engine->d_mol_counts, 0, num_boxes * sizeof(int));

    cudaMalloc(&engine->d_rng_states, num_boxes * 32 * sizeof(uint64_t));
    uint64_t* host_rng = (uint64_t*)malloc(num_boxes * 32 * sizeof(uint64_t));
    assert(host_rng != NULL);
    for (int i = 0; i < num_boxes * 32; ++i) {
        host_rng[i] = seed + (uint64_t)i * 0x9e3779b97f4a7c15ULL;
    }
    cudaMemcpy(engine->d_rng_states, host_rng, num_boxes * 32 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    free(host_rng);

    cudaMalloc(&engine->d_rho_k_re, num_boxes * CUDA_MAX_K * sizeof(float));
    cudaMalloc(&engine->d_rho_k_im, num_boxes * CUDA_MAX_K * sizeof(float));
    cudaMalloc(&engine->d_energies, num_boxes * sizeof(float));

    return engine;
}

void cuda_batch_destroy(CUDABatchEngine* engine) {
    assert(engine != NULL);
    cudaFree(engine->d_configs);
    cudaFree(engine->d_positions);
    cudaFree(engine->d_species);
    cudaFree(engine->d_mol_counts);
    cudaFree(engine->d_rng_states);
    cudaFree(engine->d_rho_k_re);
    cudaFree(engine->d_rho_k_im);
    cudaFree(engine->d_energies);
    free(engine);
}

void cuda_batch_run_steps(CUDABatchEngine* engine, int steps) {
    assert(engine != NULL);
    int chunk = MIN(steps, 10000);
    int remaining = steps;
    while (remaining > 0) {
        int cur_steps = MIN(remaining, chunk);
        batch_gcmc_kernel<<<engine->num_boxes, 32>>>(
            engine->num_boxes,
            cur_steps,
            engine->d_configs,
            engine->d_positions,
            engine->d_species,
            engine->d_mol_counts,
            engine->d_rng_states,
            engine->d_rho_k_re,
            engine->d_rho_k_im,
            engine->d_energies
        );
        cudaDeviceSynchronize();
        remaining -= cur_steps;
    }
}

void cuda_batch_get_counts(CUDABatchEngine* engine, int* counts_out) {
    assert(engine != NULL && counts_out != NULL);
    cudaMemcpy(counts_out, engine->d_mol_counts, engine->num_boxes * sizeof(int), cudaMemcpyDeviceToHost);
}

void cuda_batch_get_energies(CUDABatchEngine* engine, float* energies_out) {
    assert(engine != NULL && energies_out != NULL);
    cudaMemcpy(energies_out, engine->d_energies, engine->num_boxes * sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_batch_get_positions(CUDABatchEngine* engine, int box_idx, CUDAVec3* pos_out, int* count_out) {
    assert(engine != NULL && box_idx >= 0 && box_idx < engine->num_boxes);
    int count = 0;
    cudaMemcpy(&count, engine->d_mol_counts + box_idx, sizeof(int), cudaMemcpyDeviceToHost);
    if (count_out) *count_out = count;
    if (pos_out && count > 0) {
        cudaMemcpy(pos_out, engine->d_positions + box_idx * CUDA_MAX_MOLS * CUDA_MAX_SITES, count * sizeof(CUDAVec3), cudaMemcpyDeviceToHost);
    }
}

} // extern "C"
