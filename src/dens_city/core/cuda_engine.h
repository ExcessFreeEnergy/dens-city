#ifndef DENS_CITY_CUDA_ENGINE_H
#define DENS_CITY_CUDA_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_MAX_MOLS 2048
#define CUDA_MAX_SITES 3
#define CUDA_MAX_K 256

typedef struct CUDAVec3 {
    float x, y, z;
} CUDAVec3;

typedef struct CUDAEwaldKVector {
    CUDAVec3 k;
    float k_sq;
    float weight;
} CUDAEwaldKVector;

typedef struct CUDAPairPotential {
    int kind;
    float epsilon_lj;
    float sigma_lj;
    float rc;
    float epsilon_c;
    float q1, q2;
    float kappa_inv;
    float diameter;
    float prefactor;
    float shift_lj;
    float A_ij;
    float B_ij;
    float C_ij;
    float sigma_gauss_sq;
} CUDAPairPotential;

typedef struct CUDAExternalPotential {
    int kind;
    float low, high, width, L;
    float epsilon, sigma, cutoff, shift, q;
    float A1, A2, A3, A4;
    float phi1, phi2, phi3, phi4;
} CUDAExternalPotential;

typedef struct CUDABoxConfig {
    float T;
    float beta;
    float mu1, mu2;
    float box_x, box_y, box_z;
    float maxdispl, maxrot;
    float bond_length;
    int max_molecules;
    int mol_type;
    int electrostatics_mode;
    float ewald_alpha;
    int num_k_vectors;
    CUDAEwaldKVector k_vectors[CUDA_MAX_K];
    CUDAPairPotential pair_potentials[4][4];
    CUDAExternalPotential external_potentials[4];
} CUDABoxConfig;

typedef struct CUDABatchEngine {
    int num_boxes;
    CUDABoxConfig* d_configs;
    CUDAVec3* d_positions;
    int* d_species;
    int* d_mol_counts;
    uint64_t* d_rng_states;
    float* d_rho_k_re;
    float* d_rho_k_im;
    float* d_energies;
} CUDABatchEngine;

CUDABatchEngine* cuda_batch_create(int num_boxes, const CUDABoxConfig* configs, uint64_t seed);
void cuda_batch_destroy(CUDABatchEngine* engine);
void cuda_batch_run_steps(CUDABatchEngine* engine, int steps);
void cuda_batch_get_counts(CUDABatchEngine* engine, int* counts_out);
void cuda_batch_get_energies(CUDABatchEngine* engine, float* energies_out);
void cuda_batch_get_positions(CUDABatchEngine* engine, int box_idx, CUDAVec3* pos_out, int* count_out);

#ifdef __cplusplus
}
#endif

#endif // DENS_CITY_CUDA_ENGINE_H
