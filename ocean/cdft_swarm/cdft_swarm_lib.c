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
float env_get_p_wall(Env* env) { return env ? env->cdft_res.p_wall_bar : 0.0f; }
float env_get_omega_solv(Env* env) { return env ? env->cdft_res.omega_solv_kcal : 0.0f; }
int env_get_converged(Env* env) { return env ? env->cdft_res.converged : 0; }
float env_get_molecular_weight(Env* env) { return env ? env->mechanics.molecular_weight : 0.0f; }
float env_get_rotatable_fraction(Env* env) { return env ? env->mechanics.rotatable_bond_fraction : 0.0f; }
float env_get_aromatic_density(Env* env) { return env ? env->mechanics.aromatic_density : 0.0f; }
float env_get_pmi_linearity(Env* env) { return env ? env->mechanics.pmi_linearity : 0.0f; }
int env_get_hbd_count(Env* env) { return env ? env->mechanics.hbd_count : 0; }
int env_get_hba_count(Env* env) { return env ? env->mechanics.hba_count : 0; }

// Molecular Graph getters
int env_get_num_atoms(Env* env) { return env ? env->graph.num_atoms : 0; }
int env_get_num_bonds(Env* env) { return env ? env->graph.num_bonds : 0; }
int env_get_num_ports(Env* env) { return env ? env->graph.num_ports : 0; }

float env_get_atom_pos_x(Env* env, int i) { return env ? env->graph.atoms[i].pos.x : 0.0f; }
float env_get_atom_pos_y(Env* env, int i) { return env ? env->graph.atoms[i].pos.y : 0.0f; }
float env_get_atom_pos_z(Env* env, int i) { return env ? env->graph.atoms[i].pos.z : 0.0f; }
int env_get_atom_z(Env* env, int i) { return env ? env->graph.atoms[i].atomic_number : 0; }
int env_get_atom_is_aromatic(Env* env, int i) { return env ? (env->graph.atoms[i].is_aromatic ? 1 : 0) : 0; }
float env_get_atom_charge(Env* env, int i) { return env ? env->graph.atoms[i].charge : 0.0f; }

int env_get_bond_u(Env* env, int b) { return env ? env->graph.bonds[b].atom_u : 0; }
int env_get_bond_v(Env* env, int b) { return env ? env->graph.bonds[b].atom_v : 0; }
int env_get_bond_order(Env* env, int b) { return env ? env->graph.bonds[b].bond_order : 0; }
