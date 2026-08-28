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
float env_get_omega_solv(Env* env) {
    if (!env) return 0.0f;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_cdft_res.omega_solv_kcal : env->cdft_res.omega_solv_kcal;
}
int env_get_converged(Env* env) {
    if (!env) return 0;
    return (env->terminal_graph.num_atoms > 0) ? env->terminal_cdft_res.converged : env->cdft_res.converged;
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
