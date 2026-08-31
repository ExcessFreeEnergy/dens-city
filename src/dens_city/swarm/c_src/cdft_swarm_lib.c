#define NO_RAYLIB
#include "cdft_swarm.h"

// -------------------------------------------------------------
// Lifecycle & Buffer Management
// -------------------------------------------------------------
Env* env_create(unsigned int seed) {
    Env* env = (Env*)aligned_alloc(64, sizeof(Env));
    if (!env) return NULL;
    memset(env, 0, sizeof(Env));
    env->num_agents = 1;
    env->rng = seed;

    env->observations = (float*)aligned_alloc(64, TOTAL_OBS_SIZE * sizeof(float));
    env->actions = (float*)aligned_alloc(64, 2 * sizeof(float));
    env->rewards = (float*)aligned_alloc(64, 1 * sizeof(float));
    env->terminals = (unsigned char*)aligned_alloc(64, 1 * sizeof(unsigned char));
    env->action_mask = (unsigned char*)aligned_alloc(64, TOTAL_ACTION_MASK_SIZE * sizeof(unsigned char));

    memset(env->observations, 0, TOTAL_OBS_SIZE * sizeof(float));
    memset(env->actions, 0, 2 * sizeof(float));
    memset(env->rewards, 0, 1 * sizeof(float));
    memset(env->terminals, 0, 1 * sizeof(unsigned char));
    memset(env->action_mask, 0, TOTAL_ACTION_MASK_SIZE * sizeof(unsigned char));

    return env;
}

void env_free(Env* env) {
    if (!env) return;
    if (env->observations) free(env->observations);
    if (env->actions) free(env->actions);
    if (env->rewards) free(env->rewards);
    if (env->terminals) free(env->terminals);
    if (env->action_mask) free(env->action_mask);
    free(env);
}

void env_set_targets(Env* env, float elast, float tens, float tough, float light, float max_solv, float min_pwall, float max_mw, int min_val) {
    if (!env) return;
    env->targets.target_elasticity = elast;
    env->targets.target_tensile = tens;
    env->targets.target_toughness = tough;
    env->targets.target_lightweight = light;
    env->targets.max_solvation_kcal = max_solv;
    env->targets.min_wall_pressure_bar = min_pwall;
    env->targets.max_molecular_weight = max_mw;
    env->targets.min_valency = min_val;
}

void env_reset(Env* env) {
    if (!env) return;
    c_reset(env);
}

void env_step(Env* env, float port_action, float frag_action) {
    if (!env) return;
    env->actions[0] = port_action;
    env->actions[1] = frag_action;
    c_step(env);
}

float* env_get_observations(Env* env) { return env ? env->observations : NULL; }
float* env_get_rewards(Env* env) { return env ? env->rewards : NULL; }
unsigned char* env_get_terminals(Env* env) { return env ? env->terminals : NULL; }
unsigned char* env_get_action_mask(Env* env) {
    if (!env) return NULL;
    compute_action_mask(env);
    return env->action_mask;
}

// Telemetry getters
float env_get_p_wall(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_cdft_res.p_wall_bar : env->cdft_res.p_wall_bar;
}
float env_get_contact_ratio(Env* env) {
    if (!env) return 1.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_cdft_res.contact_ratio : env->cdft_res.contact_ratio;
}
float env_get_omega_solv(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_cdft_res.omega_solv_kcal : env->cdft_res.omega_solv_kcal;
}
int env_get_converged(Env* env) {
    if (!env) return 0;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_cdft_res.converged : env->cdft_res.converged;
}
uint64_t env_get_wl_hash(Env* env) {
    if (!env) return 0;
    MolecularGraph* g = (env->terminal_graph.num_atoms > 0) ? &env->terminal_graph : &env->graph;
    return g ? compute_wl_graph_hash(g) : 0;
}
float env_get_molecular_weight(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_mechanics.molecular_weight : env->mechanics.molecular_weight;
}
float env_get_rotatable_fraction(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_mechanics.rotatable_bond_fraction : env->mechanics.rotatable_bond_fraction;
}
float env_get_aromatic_density(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_mechanics.aromatic_density : env->mechanics.aromatic_density;
}
float env_get_pmi_linearity(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_mechanics.pmi_linearity : env->mechanics.pmi_linearity;
}
int env_get_hbd_count(Env* env) {
    if (!env) return 0;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_mechanics.hbd_count : env->mechanics.hbd_count;
}
int env_get_hba_count(Env* env) {
    if (!env) return 0;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_mechanics.hba_count : env->mechanics.hba_count;
}

// Molecular Graph getters
static inline MolecularGraph* get_active_graph(Env* env) {
    if (!env) return NULL;
    return (env->terminal_graph.num_atoms > 0) ? &env->terminal_graph : &env->graph;
}

int env_get_num_atoms(Env* env) { MolecularGraph* g = get_active_graph(env); return g ? g->num_atoms : 0; }
int env_get_num_bonds(Env* env) { MolecularGraph* g = get_active_graph(env); return g ? g->num_bonds : 0; }
int env_get_num_ports(Env* env) { MolecularGraph* g = get_active_graph(env); return g ? g->num_ports : 0; }

float env_get_atom_pos_x(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].pos.x : 0.0f; }
float env_get_atom_pos_y(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].pos.y : 0.0f; }
float env_get_atom_pos_z(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].pos.z : 0.0f; }
int env_get_atom_z(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].atomic_number : 0; }
int env_get_atom_is_aromatic(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? (g->atoms[i].is_aromatic ? 1 : 0) : 0; }
float env_get_atom_charge(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].charge : 0.0f; }
float env_get_atom_sigma(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].sigma : 3.4f; }
float env_get_atom_epsilon_k(Env* env, int i) { MolecularGraph* g = get_active_graph(env); return g ? g->atoms[i].epsilon_k : 120.0f; }

int env_get_atoms_block(Env* env, float* coords_out, float* sigmas_out, float* epsilons_out, float* charges_out, int* z_out) {
    MolecularGraph* g = get_active_graph(env);
    if (!g) return 0;
    int n = g->num_atoms;
    for (int i = 0; i < n; i++) {
        coords_out[i * 3 + 0] = g->atoms[i].pos.x;
        coords_out[i * 3 + 1] = g->atoms[i].pos.y;
        coords_out[i * 3 + 2] = g->atoms[i].pos.z;
        sigmas_out[i] = g->atoms[i].sigma;
        epsilons_out[i] = g->atoms[i].epsilon_k;
        charges_out[i] = g->atoms[i].charge;
        z_out[i] = g->atoms[i].atomic_number;
    }
    return n;
}

int env_get_bond_u(Env* env, int b) { MolecularGraph* g = get_active_graph(env); return g ? g->bonds[b].atom_u : 0; }
int env_get_bond_v(Env* env, int b) { MolecularGraph* g = get_active_graph(env); return g ? g->bonds[b].atom_v : 0; }
int env_get_bond_order(Env* env, int b) { MolecularGraph* g = get_active_graph(env); return g ? g->bonds[b].bond_order : 0; }

int env_get_atom_exclusions(Env* env, float* excl_out) {
    MolecularGraph* g = get_active_graph(env);
    if (!g) return 0;
    compute_12_13_exclusions(g, excl_out);
    return g->num_atoms;
}

// -------------------------------------------------------------
// Vectorized Multi-Environment Batch Engine (PufferLib OpenMP)
// -------------------------------------------------------------
#include <omp.h>

typedef struct {
    int num_envs;
    Env** envs;
    float* obs;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* action_masks;
    float* p_walls;
    float* contact_ratios;
    float* omega_solvs;
    float* molecular_weights;
    float* rotatable_fractions;
    float* pmi_linearities;
    float* sa_scores;
    float* r_sa_penalties;
    int* converged;
    uint64_t* wl_hashes;
} VecSwarm;

VecSwarm* vec_swarm_create(int num_envs, unsigned int seed) {
    VecSwarm* v = (VecSwarm*)malloc(sizeof(VecSwarm));
    if (!v) return NULL;
    v->num_envs = num_envs;
    v->envs = (Env**)malloc(num_envs * sizeof(Env*));
    v->obs = (float*)aligned_alloc(64, (size_t)num_envs * TOTAL_OBS_SIZE * sizeof(float));
    v->actions = (float*)aligned_alloc(64, (size_t)num_envs * 2 * sizeof(float));
    v->rewards = (float*)aligned_alloc(64, (size_t)num_envs * 1 * sizeof(float));
    v->terminals = (unsigned char*)aligned_alloc(64, (size_t)num_envs * 1 * sizeof(unsigned char));
    v->action_masks = (unsigned char*)aligned_alloc(64, (size_t)num_envs * TOTAL_ACTION_MASK_SIZE * sizeof(unsigned char));
    v->p_walls = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->contact_ratios = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->omega_solvs = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->molecular_weights = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->rotatable_fractions = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->pmi_linearities = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->sa_scores = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->r_sa_penalties = (float*)aligned_alloc(64, (size_t)num_envs * sizeof(float));
    v->converged = (int*)aligned_alloc(64, (size_t)num_envs * sizeof(int));
    v->wl_hashes = (uint64_t*)aligned_alloc(64, (size_t)num_envs * sizeof(uint64_t));

    memset(v->obs, 0, (size_t)num_envs * TOTAL_OBS_SIZE * sizeof(float));
    memset(v->actions, 0, (size_t)num_envs * 2 * sizeof(float));
    memset(v->rewards, 0, (size_t)num_envs * 1 * sizeof(float));
    memset(v->terminals, 0, (size_t)num_envs * 1 * sizeof(unsigned char));
    memset(v->action_masks, 0, (size_t)num_envs * TOTAL_ACTION_MASK_SIZE * sizeof(unsigned char));
    memset(v->p_walls, 0, (size_t)num_envs * sizeof(float));
    memset(v->contact_ratios, 0, (size_t)num_envs * sizeof(float));
    memset(v->omega_solvs, 0, (size_t)num_envs * sizeof(float));
    memset(v->molecular_weights, 0, (size_t)num_envs * sizeof(float));
    memset(v->rotatable_fractions, 0, (size_t)num_envs * sizeof(float));
    memset(v->pmi_linearities, 0, (size_t)num_envs * sizeof(float));
    memset(v->sa_scores, 0, (size_t)num_envs * sizeof(float));
    memset(v->r_sa_penalties, 0, (size_t)num_envs * sizeof(float));
    memset(v->converged, 0, (size_t)num_envs * sizeof(int));
    memset(v->wl_hashes, 0, (size_t)num_envs * sizeof(uint64_t));

    for (int i = 0; i < num_envs; i++) {
        v->envs[i] = (Env*)aligned_alloc(64, sizeof(Env));
        memset(v->envs[i], 0, sizeof(Env));
        v->envs[i]->num_agents = 1;
        v->envs[i]->rng = seed + (unsigned int)i * 1000 + 1;
        v->envs[i]->observations = &v->obs[i * TOTAL_OBS_SIZE];
        v->envs[i]->actions = &v->actions[i * 2];
        v->envs[i]->rewards = &v->rewards[i];
        v->envs[i]->terminals = &v->terminals[i];
        v->envs[i]->action_mask = &v->action_masks[i * TOTAL_ACTION_MASK_SIZE];
        c_reset(v->envs[i]);
    }
    return v;
}

void vec_swarm_set_targets(VecSwarm* v, float elast, float tens, float tough, float light, float max_solv, float min_pwall, float max_mw, int min_val, float sa_thresh, float sa_slope) {
    if (!v) return;
    for (int i = 0; i < v->num_envs; i++) {
        v->envs[i]->targets.target_elasticity = elast;
        v->envs[i]->targets.target_tensile = tens;
        v->envs[i]->targets.target_toughness = tough;
        v->envs[i]->targets.target_lightweight = light;
        v->envs[i]->targets.max_solvation_kcal = max_solv;
        v->envs[i]->targets.min_wall_pressure_bar = min_pwall;
        v->envs[i]->targets.max_molecular_weight = max_mw;
        v->envs[i]->targets.min_valency = min_val;
        v->envs[i]->targets.sa_threshold = sa_thresh;
        v->envs[i]->targets.sa_penalty_slope = sa_slope;
    }
}

void vec_swarm_reset(VecSwarm* v) {
    if (!v) return;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v->num_envs; i++) {
        c_reset(v->envs[i]);
    }
}

void vec_swarm_step(VecSwarm* v) {
    if (!v) return;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < v->num_envs; i++) {
        c_step(v->envs[i]);
        Env* env = v->envs[i];
        if (env->terminal_graph.num_atoms > 0) {
            v->p_walls[i] = env->terminal_cdft_res.p_wall_bar;
            v->contact_ratios[i] = env->terminal_cdft_res.contact_ratio;
            v->omega_solvs[i] = env->terminal_cdft_res.omega_solv_kcal;
            v->molecular_weights[i] = env->terminal_mechanics.molecular_weight;
            v->rotatable_fractions[i] = env->terminal_mechanics.rotatable_bond_fraction;
            v->pmi_linearities[i] = env->terminal_mechanics.pmi_linearity;
            v->sa_scores[i] = env->terminal_mechanics.sa_score;
            v->wl_hashes[i] = compute_wl_graph_hash(&env->terminal_graph);
            float r_sa = 0.0f;
            if (env->targets.sa_threshold > 0.0f && env->terminal_mechanics.sa_score > env->targets.sa_threshold) {
                float slope = (env->targets.sa_penalty_slope > 0.0f) ? env->targets.sa_penalty_slope : 2.0f;
                r_sa = -slope * (env->terminal_mechanics.sa_score - env->targets.sa_threshold);
            }
            v->r_sa_penalties[i] = r_sa;
            v->converged[i] = env->terminal_cdft_res.converged;
        } else {
            v->p_walls[i] = env->cdft_res.p_wall_bar;
            v->contact_ratios[i] = env->cdft_res.contact_ratio;
            v->omega_solvs[i] = env->cdft_res.omega_solv_kcal;
            v->molecular_weights[i] = env->mechanics.molecular_weight;
            v->rotatable_fractions[i] = env->mechanics.rotatable_bond_fraction;
            v->pmi_linearities[i] = env->mechanics.pmi_linearity;
            v->sa_scores[i] = env->mechanics.sa_score;
            v->wl_hashes[i] = compute_wl_graph_hash(&env->graph);
            float r_sa = 0.0f;
            if (env->targets.sa_threshold > 0.0f && env->mechanics.sa_score > env->targets.sa_threshold) {
                float slope = (env->targets.sa_penalty_slope > 0.0f) ? env->targets.sa_penalty_slope : 2.0f;
                r_sa = -slope * (env->mechanics.sa_score - env->targets.sa_threshold);
            }
            v->r_sa_penalties[i] = r_sa;
            v->converged[i] = env->cdft_res.converged;
        }
    }
}

void vec_swarm_free(VecSwarm* v) {
    if (!v) return;
    for (int i = 0; i < v->num_envs; i++) {
        if (v->envs[i]) free(v->envs[i]);
    }
    if (v->envs) free(v->envs);
    if (v->obs) free(v->obs);
    if (v->actions) free(v->actions);
    if (v->rewards) free(v->rewards);
    if (v->terminals) free(v->terminals);
    if (v->action_masks) free(v->action_masks);
    if (v->p_walls) free(v->p_walls);
    if (v->contact_ratios) free(v->contact_ratios);
    if (v->omega_solvs) free(v->omega_solvs);
    if (v->molecular_weights) free(v->molecular_weights);
    if (v->rotatable_fractions) free(v->rotatable_fractions);
    if (v->pmi_linearities) free(v->pmi_linearities);
    if (v->sa_scores) free(v->sa_scores);
    if (v->r_sa_penalties) free(v->r_sa_penalties);
    if (v->converged) free(v->converged);
    if (v->wl_hashes) free(v->wl_hashes);
    free(v);
}

float* vec_swarm_get_obs(VecSwarm* v) { return v ? v->obs : NULL; }
float* vec_swarm_get_actions(VecSwarm* v) { return v ? v->actions : NULL; }
float* vec_swarm_get_rewards(VecSwarm* v) { return v ? v->rewards : NULL; }
unsigned char* vec_swarm_get_terminals(VecSwarm* v) { return v ? v->terminals : NULL; }
unsigned char* vec_swarm_get_action_masks(VecSwarm* v) { return v ? v->action_masks : NULL; }
float* vec_swarm_get_p_walls(VecSwarm* v) { return v ? v->p_walls : NULL; }
float* vec_swarm_get_contact_ratios(VecSwarm* v) { return v ? v->contact_ratios : NULL; }
float* vec_swarm_get_omega_solvs(VecSwarm* v) { return v ? v->omega_solvs : NULL; }
float* vec_swarm_get_molecular_weights(VecSwarm* v) { return v ? v->molecular_weights : NULL; }
float* vec_swarm_get_rotatable_fractions(VecSwarm* v) { return v ? v->rotatable_fractions : NULL; }
float* vec_swarm_get_pmi_linearities(VecSwarm* v) { return v ? v->pmi_linearities : NULL; }
float* vec_swarm_get_sa_scores(VecSwarm* v) { return v ? v->sa_scores : NULL; }
float* vec_swarm_get_r_sa_penalties(VecSwarm* v) { return v ? v->r_sa_penalties : NULL; }
int* vec_swarm_get_converged(VecSwarm* v) { return v ? v->converged : NULL; }
uint64_t* vec_swarm_get_wl_hashes(VecSwarm* v) { return v ? v->wl_hashes : NULL; }
Env* vec_swarm_get_env_ptr(VecSwarm* v, int i) { return (v && i >= 0 && i < v->num_envs) ? v->envs[i] : NULL; }

