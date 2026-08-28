#ifndef CDFT_SWARM_H
#define CDFT_SWARM_H

#include "mol_graph.h"
#include "mechanics_eval.h"
#include "cdft_solver.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#ifndef NO_RAYLIB
#include "raylib.h"
#endif

#define MAX_EPISODE_STEPS 16
#define NUM_FRAGMENT_CHOICES 12 // 0..11 fragments in library
#define ACTION_FINALIZE 12      // Index 12 indicates molecule finalization / stop

#define OBS_GRAPH_FEATS 16
#define OBS_PORT_FEATS (MAX_PORTS * 4) // 16 * 4 = 64
#define OBS_TARGET_FEATS 8
#define TOTAL_OBS_SIZE (OBS_GRAPH_FEATS + OBS_PORT_FEATS + OBS_TARGET_FEATS) // 16 + 64 + 8 = 88

#define TOTAL_ACTION_MASK_SIZE (MAX_PORTS + NUM_FRAGMENT_CHOICES + 1) // 16 + 13 = 29

// Multi-Objective Target Specification from YAML
typedef struct {
    float target_elasticity;     // Target weight for springiness / rotatable bonds [0.0, 1.0]
    float target_tensile;        // Target weight for yield stress / aromatic / PMI [0.0, 1.0]
    float target_toughness;      // Target weight for cracking resistance / H-bonds [0.0, 1.0]
    float target_lightweight;    // Target weight for low mass / high FFV [0.0, 1.0]
    float max_solvation_kcal;    // Target upper bound for solvation free energy (e.g. -5.0 kcal/mol)
    float min_wall_pressure_bar; // Target lower bound for wall contact pressure (e.g. 10.0 bar)
    float max_molecular_weight;  // Maximum molecular weight ceiling (e.g. 850.0 amu)
    int min_valency;             // Minimum reactive crosslinking sites required (e.g. 2)
    float sa_threshold;          // SA score penalty threshold (e.g. 4.5)
    float sa_penalty_slope;      // SA score excess penalty slope (e.g. 2.0)
} TargetSpec;

// Required PufferLib Log struct (all floats, n last)
typedef struct {
    float perf;
    float score;
    float r_thermo;
    float r_elasticity;
    float r_tensile;
    float r_toughness;
    float r_lightweight;
    float p_wall;
    float omega_solv;
    float valid_molecules;
    float sa_score;
    float r_sa_penalty;
    float n; // Required as last field
} Log;

#ifndef NO_RAYLIB
// Raylib Client visualizer state
typedef struct {
    Camera3D camera;
    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;
} Client;
#endif

// PufferLib Environment Struct
typedef struct CDFT_Swarm_Env {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* action_mask;
    int num_agents;
    unsigned int rng;

    // Molecular graph & state
    MolecularGraph graph;
    MechanicsProfile mechanics;
    FluidCDFTParams params;
    CDFT_Env_Context ctx;
    CDFT_Result cdft_res;

    // Completed episode snapshots
    MolecularGraph terminal_graph;
    MechanicsProfile terminal_mechanics;
    CDFT_Result terminal_cdft_res;

    TargetSpec targets;
    int step_count;
    int current_scaffold_idx;

    void* client;
} CDFT_Swarm_Env;

#define Env CDFT_Swarm_Env

// -------------------------------------------------------------
// Action Masking Computation (MY_ACTION_MASK)
// -------------------------------------------------------------
static inline void compute_action_mask(Env* env) {
    if (!env->action_mask) return;
    memset(env->action_mask, 0, TOTAL_ACTION_MASK_SIZE);

    int empty_port_count = 0;
    bool port_has_valid_frag[MAX_PORTS] = {false};
    bool frag_has_valid_port[NUM_FRAGMENT_CHOICES] = {false};

    // Precompute 128-bit adjacency masks for all atoms in graph
    Bitmask128 adj[MAX_ATOMS];
    compute_graph_adjacency_128(&env->graph, adj);

    // Evaluate pair-wise feasibility across all open ports and fragment choices
    for (int p = 0; p < env->graph.num_ports; p++) {
        if (env->graph.ports[p].state != PORT_STATE_EMPTY) continue;
        empty_port_count++;
        int origin = env->graph.ports[p].origin_atom;
        const Bitmask128* origin_adj = (origin >= 0 && origin < MAX_ATOMS) ? &adj[origin] : NULL;

        for (int f = 0; f < NUM_FRAGMENT_CHOICES; f++) {
            if (is_attachment_valid_with_adj(&env->graph, p, f, env->targets.max_molecular_weight, NULL, NULL, NULL, origin_adj)) {
                port_has_valid_frag[p] = true;
                frag_has_valid_port[f] = true;
            }
        }
    }

    // 1. Port action mask (indices 0..15): 1 if port is empty and has at least one valid fragment attachment
    for (int p = 0; p < MAX_PORTS; p++) {
        if (p < env->graph.num_ports && port_has_valid_frag[p]) {
            env->action_mask[p] = 1;
        } else {
            env->action_mask[p] = 0;
        }
    }

    // 2. Fragment choice action mask (indices 16..27 for fragments 0..11)
    unsigned char* frag_mask = &env->action_mask[MAX_PORTS];
    for (int f = 0; f < NUM_FRAGMENT_CHOICES; f++) {
        frag_mask[f] = frag_has_valid_port[f] ? 1 : 0;
    }

    // 3. Finalize action mask (index 28)
    // Allowed ONLY if the molecule has grown past bare scaffold (>=16 atoms, >=2 attached fragments, MW >= 180)
    // and has at least min_valency open reactive ports
    if (empty_port_count >= env->targets.min_valency &&
        env->graph.num_atoms >= 16 &&
        env->graph.num_attached_fragments >= 2 &&
        env->mechanics.molecular_weight >= 180.0f) {
        frag_mask[NUM_FRAGMENT_CHOICES] = 1; // Can finalize
    } else {
        frag_mask[NUM_FRAGMENT_CHOICES] = 0;
    }
}

// -------------------------------------------------------------
// Observation Builder
// -------------------------------------------------------------
static inline void compute_observations(Env* env) {
    float* obs = env->observations;
    int idx = 0;

    // 1. Molecular Graph Features (16 floats)
    obs[idx++] = (float)env->graph.num_atoms / (float)MAX_ATOMS;
    obs[idx++] = (float)env->graph.num_bonds / (float)MAX_BONDS;
    obs[idx++] = (float)env->graph.num_ports / (float)MAX_PORTS;
    obs[idx++] = env->mechanics.molecular_weight / env->targets.max_molecular_weight;
    obs[idx++] = env->mechanics.rotatable_bond_fraction;
    obs[idx++] = env->mechanics.aromatic_density;
    obs[idx++] = env->mechanics.pmi_linearity;
    obs[idx++] = env->mechanics.kuhn_persistence_proxy;
    obs[idx++] = (float)env->mechanics.hbd_count / 10.0f;
    obs[idx++] = (float)env->mechanics.hba_count / 10.0f;
    obs[idx++] = env->mechanics.fractional_free_volume;
    obs[idx++] = env->mechanics.tpsa_proxy / 200.0f;
    obs[idx++] = (float)env->mechanics.multivalency_count / 8.0f;
    obs[idx++] = (float)env->step_count / (float)MAX_EPISODE_STEPS;
    obs[idx++] = env->cdft_res.converged ? 1.0f : 0.0f;
    obs[idx++] = env->cdft_res.p_wall_bar / 100.0f;

    // 2. Open Port Geometric Vectors (16 * 4 = 64 floats)
    for (int p = 0; p < MAX_PORTS; p++) {
        if (p < env->graph.num_ports) {
            obs[idx++] = env->graph.ports[p].normal.x;
            obs[idx++] = env->graph.ports[p].normal.y;
            obs[idx++] = env->graph.ports[p].normal.z;
            obs[idx++] = (env->graph.ports[p].state == PORT_STATE_EMPTY) ? 1.0f : 0.0f;
        } else {
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
            obs[idx++] = 0.0f;
        }
    }

    // 3. YAML Target Objective Vector (8 floats)
    obs[idx++] = env->targets.target_elasticity;
    obs[idx++] = env->targets.target_tensile;
    obs[idx++] = env->targets.target_toughness;
    obs[idx++] = env->targets.target_lightweight;
    obs[idx++] = env->targets.max_solvation_kcal;
    obs[idx++] = env->targets.min_wall_pressure_bar / 100.0f;
    obs[idx++] = env->targets.max_molecular_weight / 1000.0f;
    obs[idx++] = (float)env->targets.min_valency / 4.0f;
}

// -------------------------------------------------------------
// c_reset
// -------------------------------------------------------------
static inline void c_reset(Env* env) {
    env->step_count = 0;

    // Choose starting scaffold (0: Benzene, 1: Triphenylamine, 2: Adamantane)
    env->current_scaffold_idx = rand_r(&env->rng) % 3;
    init_base_scaffold(&env->graph, env->current_scaffold_idx);
    env->mechanics = evaluate_mechanics(&env->graph);

    // If targets are unset, assign balanced default targets
    if (env->targets.max_molecular_weight <= 0.0f) {
        env->targets.target_elasticity = 0.35f;
        env->targets.target_tensile = 0.35f;
        env->targets.target_toughness = 0.20f;
        env->targets.target_lightweight = 0.10f;
        env->targets.max_solvation_kcal = -3.0f;
        env->targets.min_wall_pressure_bar = 15.0f;
        env->targets.max_molecular_weight = 850.0f;
        env->targets.min_valency = 2;
        env->targets.sa_threshold = 4.5f;
        env->targets.sa_penalty_slope = 2.0f;
    }

    compute_observations(env);
    compute_action_mask(env);
}

// -------------------------------------------------------------
// c_step
// -------------------------------------------------------------
static inline void c_step(Env* env) {
    env->step_count++;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0;

    int selected_port = (int)env->actions[0];
    int selected_frag = (int)env->actions[1];

    bool final_step = false;

    // Check if agent triggered FINALIZE or hit step limit
    if (selected_frag >= ACTION_FINALIZE || env->step_count >= MAX_EPISODE_STEPS) {
        final_step = true;
    } else {
        // Execute SE(3) Rigid-Body Attachment
        bool attached = rigid_body_attach(&env->graph, selected_port, selected_frag);
        if (!attached) {
            // Invalid attachment attempted (penalize slightly)
            env->rewards[0] -= 0.10f;
        } else {
            // Episodic objective evaluation: zero intermediate cookie crumbs
            env->rewards[0] = 0.0f;
        }

        // Check if all ports are now filled
        int empty_ports = 0;
        for (int p = 0; p < env->graph.num_ports; p++) {
            if (env->graph.ports[p].state == PORT_STATE_EMPTY) empty_ports++;
        }
        if (empty_ports == 0) {
            final_step = true;
        }
    }

    // Evaluate Mechanics
    env->mechanics = evaluate_mechanics(&env->graph);

    // Final Evaluation & Dual Reward
    if (final_step) {
        env->terminals[0] = 1;
        env->log.n += 1.0f;

        // 1. Solve 1D cDFT Thermodynamics
        derive_fluid_parameters_from_graph(&env->graph, &env->params);
        init_cdft_context(&env->ctx, &env->params);
        env->cdft_res = solve_cdft_pufferlib_step(&env->ctx, &env->params);

        // 2. Compute Dual Reward Components
        // A. Thermodynamic Survival Check (Asymptotically Bounded via tanh)
        float r_thermo = 0.0f;
        if (!env->cdft_res.converged) {
            r_thermo = -5.0f; // Harsh penalty for unphysical non-convergence
        } else {
            // Solvation check: bounded bonus in [-3.0, +2.0]
            float solv_diff = env->targets.max_solvation_kcal - env->cdft_res.omega_solv_kcal;
            if (solv_diff >= 0.0f) {
                r_thermo += 2.0f * tanhf(0.1f * solv_diff);
            } else {
                r_thermo -= 1.0f + 2.0f * tanhf(0.1f * fabsf(solv_diff));
            }

            // Wall wetting contact pressure check: bounded bonus in [-3.0, +2.0]
            float p_diff = env->cdft_res.p_wall_bar - env->targets.min_wall_pressure_bar;
            if (p_diff >= 0.0f) {
                r_thermo += 2.0f * tanhf(0.02f * p_diff);
            } else {
                r_thermo -= 1.0f + 2.0f * tanhf(0.05f * fabsf(p_diff));
            }
        }

        // B. Mechanical Heuristics Matches
        float r_elasticity = env->targets.target_elasticity * (
            2.0f * env->mechanics.rotatable_bond_fraction +
            1.5f * (1.0f - fminf(1.0f, env->mechanics.kuhn_persistence_proxy)) +
            0.1f * fminf(20.0f, env->mechanics.crosslink_node_distance)
        );

        float r_tensile = env->targets.target_tensile * (
            2.5f * env->mechanics.aromatic_density +
            3.0f * env->mechanics.pmi_linearity +
            1.0f * (float)fminf(4, env->mechanics.multivalency_count)
        );

        float r_toughness = env->targets.target_toughness * (
            1.5f * (float)fminf(6, env->mechanics.hbd_count + env->mechanics.hba_count) +
            2.0f * fminf(5.0f, env->mechanics.sacrificial_hbond_score) +
            1.5f * fminf(4.0f, env->mechanics.pi_pi_stacking_score)
        );

        float r_lightweight = env->targets.target_lightweight * (
            2.0f * env->mechanics.fractional_free_volume -
            1.0f * env->mechanics.heavy_atom_penalty
        );

        // Valence constraint penalty & Mechanical Obliteration
        int active_valency = 0;
        for (int p = 0; p < env->graph.num_ports; p++) {
            if (env->graph.ports[p].state == PORT_STATE_EMPTY) active_valency++;
        }

        float r_penalties = 0.0f;
        if (active_valency < env->targets.min_valency) {
            // Unreactive molecule: Explicitly obliterate all mechanical rewards
            r_elasticity = 0.0f;
            r_tensile = 0.0f;
            r_toughness = 0.0f;
            r_lightweight = 0.0f;
            r_penalties += 10.0f * (float)(env->targets.min_valency - active_valency);
        }

        // Under-size penalty for failing to grow beyond scaffold
        if (env->graph.num_atoms < 16 || env->graph.num_attached_fragments < 2) {
            r_elasticity = 0.0f;
            r_tensile = 0.0f;
            r_toughness = 0.0f;
            r_lightweight = 0.0f;
            r_penalties += 15.0f * (1.0f - (float)env->graph.num_atoms / 16.0f);
        }

        // Steep Quadratic Overweight Penalty
        if (env->mechanics.molecular_weight > env->targets.max_molecular_weight) {
            float mw_excess = env->mechanics.molecular_weight - env->targets.max_molecular_weight;
            float mw_ratio = mw_excess / env->targets.max_molecular_weight;
            r_penalties += 5.0f * mw_ratio + 15.0f * (mw_ratio * mw_ratio);
        }

        // Synthetic Accessibility (SA) Score Penalty
        float r_sa = 0.0f;
        float sa_score = env->mechanics.sa_score;
        if (env->targets.sa_threshold > 0.0f && sa_score > env->targets.sa_threshold) {
            float sa_excess = sa_score - env->targets.sa_threshold;
            float slope = (env->targets.sa_penalty_slope > 0.0f) ? env->targets.sa_penalty_slope : 2.0f;
            r_sa = -slope * sa_excess;
            r_penalties -= r_sa; // Subtracting negative penalty adds to total penalties
        }

        float total_reward = r_thermo + r_elasticity + r_tensile + r_toughness + r_lightweight - r_penalties;
        env->rewards[0] += total_reward;

        // Logging statistics
        env->log.r_thermo += r_thermo;
        env->log.r_elasticity += r_elasticity;
        env->log.r_tensile += r_tensile;
        env->log.r_toughness += r_toughness;
        env->log.r_lightweight += r_lightweight;
        env->log.p_wall = env->cdft_res.p_wall_bar;
        env->log.omega_solv = env->cdft_res.omega_solv_kcal;
        env->log.sa_score = sa_score;
        env->log.r_sa_penalty += r_sa;
        env->log.score += total_reward;
        env->log.perf += (total_reward > 0.0f) ? 1.0f : 0.0f;

        if (env->cdft_res.converged && r_penalties <= 0.0f) {
            env->log.valid_molecules += 1.0f;
        }

        env->terminal_graph = env->graph;
        env->terminal_mechanics = env->mechanics;
        env->terminal_cdft_res = env->cdft_res;
        c_reset(env);
        env->cdft_res = env->terminal_cdft_res;
        return;
    }

    compute_observations(env);
    compute_action_mask(env);
}

// -------------------------------------------------------------
// Interactive Raylib Visualizer (c_render)
// -------------------------------------------------------------
#ifndef NO_RAYLIB
static inline void init_client(Env* env) {
    if (env->client) return;
    Client* c = (Client*)calloc(1, sizeof(Client));
    c->camera.position = (Vector3){0.0f, -20.0f, 15.0f};
    c->camera.target = (Vector3){0.0f, 0.0f, 0.0f};
    c->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    c->camera.fovy = 45.0f;
    c->camera.projection = CAMERA_PERSPECTIVE;
    c->camera_distance = 25.0f;
    c->camera_azimuth = 0.0f;
    c->camera_elevation = 0.35f;
    env->client = c;
}

static inline void c_render(Env* env) {
    if (!IsWindowReady()) {
        SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE);
        InitWindow(1280, 720, "dens-city: Multi-Objective RL Molecular Swarm");
        SetTargetFPS(60);
        init_client(env);
    }

    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        CloseWindow();
        exit(0);
    }

    Client* c = env->client;
    // Mouse drag camera rotation
    Vector2 mouse_pos = GetMousePosition();
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        c->is_dragging = true;
        c->last_mouse_pos = mouse_pos;
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) c->is_dragging = false;

    if (c->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        float dx = mouse_pos.x - c->last_mouse_pos.x;
        float dy = mouse_pos.y - c->last_mouse_pos.y;
        c->camera_azimuth -= dx * 0.005f;
        c->camera_elevation = fmaxf(-1.4f, fminf(1.4f, c->camera_elevation + dy * 0.005f));
        c->last_mouse_pos = mouse_pos;
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        c->camera_distance = fmaxf(5.0f, fminf(60.0f, c->camera_distance - wheel * 2.0f));
    }

    c->camera.position.x = c->camera_distance * cosf(c->camera_elevation) * sinf(c->camera_azimuth);
    c->camera.position.y = -c->camera_distance * cosf(c->camera_elevation) * cosf(c->camera_azimuth);
    c->camera.position.z = c->camera_distance * sinf(c->camera_elevation);

    BeginDrawing();
    ClearBackground((Color){18, 22, 28, 255});

    // 1. 3D Molecular View
    BeginMode3D(c->camera);
    DrawGrid(20, 2.0f);

    // Draw Bonds
    for (int b = 0; b < env->graph.num_bonds; b++) {
        int u = env->graph.bonds[b].atom_u;
        int v = env->graph.bonds[b].atom_v;
        if (u < env->graph.num_atoms && v < env->graph.num_atoms) {
            Vector3 pu = (Vector3){env->graph.atoms[u].pos.x, env->graph.atoms[u].pos.y, env->graph.atoms[u].pos.z};
            Vector3 pv = (Vector3){env->graph.atoms[v].pos.x, env->graph.atoms[v].pos.y, env->graph.atoms[v].pos.z};
            DrawCylinderEx(pu, pv, 0.12f, 0.12f, 8, (Color){180, 180, 190, 255});
        }
    }

    // Draw Atoms (CPK Colors)
    for (int i = 0; i < env->graph.num_atoms; i++) {
        const AtomSite* a = &env->graph.atoms[i];
        Vector3 pos = (Vector3){a->pos.x, a->pos.y, a->pos.z};
        Color cpk_color = (Color){150, 150, 150, 255};
        float radius = 0.45f;

        if (a->atomic_number == 1) { cpk_color = (Color){240, 240, 240, 255}; radius = 0.28f; } // H
        else if (a->atomic_number == 6) { cpk_color = a->is_aromatic ? (Color){70, 70, 75, 255} : (Color){50, 50, 50, 255}; } // C
        else if (a->atomic_number == 7) { cpk_color = (Color){45, 105, 225, 255}; } // N (Blue)
        else if (a->atomic_number == 8) { cpk_color = (Color){220, 45, 45, 255}; }  // O (Red)
        else if (a->atomic_number == 9) { cpk_color = (Color){115, 215, 75, 255}; } // F (Green)
        else if (a->atomic_number == 16) { cpk_color = (Color){235, 205, 50, 255}; } // S (Yellow)

        DrawSphere(pos, radius, cpk_color);
    }

    // Draw Open Ports
    for (int p = 0; p < env->graph.num_ports; p++) {
        if (env->graph.ports[p].state == PORT_STATE_EMPTY) {
            Vector3 ppos = (Vector3){env->graph.ports[p].pos.x, env->graph.ports[p].pos.y, env->graph.ports[p].pos.z};
            Vector3 pend = (Vector3){
                ppos.x + env->graph.ports[p].normal.x * 1.0f,
                ppos.y + env->graph.ports[p].normal.y * 1.0f,
                ppos.z + env->graph.ports[p].normal.z * 1.0f
            };
            DrawLine3D(ppos, pend, (Color){0, 230, 230, 255});
            DrawSphere(pend, 0.15f, (Color){0, 255, 255, 255});
        }
    }
    EndMode3D();

    // 2. HUD Telemetry & Radar Dashboard
    DrawRectangle(20, 20, 360, 310, (Color){10, 14, 20, 220});
    DrawRectangleLines(20, 20, 360, 310, (Color){60, 80, 110, 255});

    DrawText("STAGE 1: CDFT SWARM AGENT", 35, 30, 16, (Color){0, 220, 220, 255});
    DrawText(TextFormat("Atoms: %d | Bonds: %d | Ports: %d", env->graph.num_atoms, env->graph.num_bonds, env->graph.num_ports), 35, 55, 14, RAYWHITE);
    DrawText(TextFormat("Molecular Weight: %.1f amu", env->mechanics.molecular_weight), 35, 75, 14, RAYWHITE);
    DrawText(TextFormat("Rotatable Fraction: %.2f", env->mechanics.rotatable_bond_fraction), 35, 95, 14, RAYWHITE);
    DrawText(TextFormat("PMI Linearity: %.2f", env->mechanics.pmi_linearity), 35, 115, 14, RAYWHITE);
    DrawText(TextFormat("H-Bond Donors/Acceptors: %d / %d", env->mechanics.hbd_count, env->mechanics.hba_count), 35, 135, 14, RAYWHITE);
    DrawText(TextFormat("Fract. Free Volume: %.2f", env->mechanics.fractional_free_volume), 35, 155, 14, RAYWHITE);

    DrawLine(35, 180, 365, 180, (Color){80, 90, 110, 255});
    DrawText("cDFT THERMODYNAMIC OBSERVABLES", 35, 190, 14, (Color){255, 180, 50, 255});
    DrawText(TextFormat("Solvation Free Energy: %.2f kcal/mol", env->cdft_res.omega_solv_kcal), 35, 210, 14, RAYWHITE);
    DrawText(TextFormat("Wall Contact Pressure: %.1f bar", env->cdft_res.p_wall_bar), 35, 230, 14, RAYWHITE);
    DrawText(TextFormat("Liquid Packing eta: %.3f", env->cdft_res.packing_fraction), 35, 250, 14, RAYWHITE);
    DrawText(TextFormat("Solver Status: %s (%d iters)", env->cdft_res.converged ? "CONVERGED" : "PENDING", env->cdft_res.iterations_taken), 35, 270, 14, env->cdft_res.converged ? GREEN : YELLOW);

    // 3. Live 1D Density Profile rho(z) Graph Overlay
    int graph_x = GetScreenWidth() - 360;
    int graph_y = 20;
    int graph_w = 340;
    int graph_h = 180;
    DrawRectangle(graph_x, graph_y, graph_w, graph_h, (Color){10, 14, 20, 220});
    DrawRectangleLines(graph_x, graph_y, graph_w, graph_h, (Color){60, 80, 110, 255});
    DrawText("1D Slit-Pore Density rho(z)", graph_x + 15, graph_y + 10, 14, (Color){0, 220, 220, 255});

    float max_plot_rho = fmaxf(0.05f, env->cdft_res.peak_density * 1.2f);
    for (int i = 0; i < N_GRID - 1; i++) {
        float x1 = graph_x + 15 + (float)i * ((float)(graph_w - 30) / (float)N_GRID);
        float x2 = graph_x + 15 + (float)(i + 1) * ((float)(graph_w - 30) / (float)N_GRID);
        float y1 = graph_y + graph_h - 20 - (env->ctx.rho[i] / max_plot_rho) * (graph_h - 50);
        float y2 = graph_y + graph_h - 20 - (env->ctx.rho[i + 1] / max_plot_rho) * (graph_h - 50);
        DrawLine((int)x1, (int)y1, (int)x2, (int)y2, (Color){0, 255, 200, 255});
    }

    EndDrawing();
}

static inline void c_close(Env* env) {
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (env->client) {
        free(env->client);
        env->client = NULL;
    }
}
#else
static inline void c_render(Env* env) { (void)env; }
static inline void c_close(Env* env) { (void)env; }
#endif

#endif // CDFT_SWARM_H
