#ifndef MOL_GRAPH_H
#define MOL_GRAPH_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

#define MAX_ATOMS 64
#define MAX_BONDS 128
#define MAX_PORTS 16
#define MAX_FRAGMENTS 16

#define PORT_STATE_EMPTY 0
#define PORT_STATE_OCCUPIED 1
#define PORT_STATE_BLOCKED 2

// -------------------------------------------------------------
// 3D Vector & Matrix Algebra for SE(3) Rigid-Body Assembly
// -------------------------------------------------------------
typedef struct {
    float x, y, z;
} Vec3;

typedef struct {
    float m[3][3];
} Mat3;

static inline Vec3 vec3_create(float x, float y, float z) {
    return (Vec3){x, y, z};
}

static inline Vec3 vec3_add(Vec3 a, Vec3 b) {
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 vec3_scale(Vec3 a, float s) {
    return (Vec3){a.x * s, a.y * s, a.z * s};
}

static inline float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return (Vec3){
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

static inline float vec3_norm(Vec3 a) {
    return sqrtf(vec3_dot(a, a));
}

static inline Vec3 vec3_normalize(Vec3 a) {
    float n = vec3_norm(a);
    if (n < 1e-7f) return (Vec3){0.0f, 0.0f, 1.0f};
    return vec3_scale(a, 1.0f / n);
}

static inline float vec3_dist(Vec3 a, Vec3 b) {
    return vec3_norm(vec3_sub(a, b));
}

static inline Vec3 mat3_vec3_mul(Mat3 m, Vec3 v) {
    return (Vec3){
        m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z,
        m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z,
        m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z
    };
}

// Compute rotation matrix R aligning unit vector src to unit vector dst (Rodrigues formula)
static inline Mat3 mat3_align_vectors(Vec3 src, Vec3 dst) {
    Vec3 v1 = vec3_normalize(src);
    Vec3 v2 = vec3_normalize(dst);
    float c = vec3_dot(v1, v2);
    Mat3 R;

    if (c > 0.999999f) {
        // Identity matrix
        memset(&R, 0, sizeof(Mat3));
        R.m[0][0] = 1.0f; R.m[1][1] = 1.0f; R.m[2][2] = 1.0f;
        return R;
    }
    if (c < -0.999999f) {
        // 180 degree rotation around an orthogonal axis
        Vec3 axis = fabsf(v1.x) < 0.8f ? vec3_create(1.0f, 0.0f, 0.0f) : vec3_create(0.0f, 1.0f, 0.0f);
        Vec3 u = vec3_normalize(vec3_cross(v1, axis));
        for (int i = 0; i < 3; i++) {
            float ui = (i == 0) ? u.x : (i == 1 ? u.y : u.z);
            for (int j = 0; j < 3; j++) {
                float uj = (j == 0) ? u.x : (j == 1 ? u.y : u.z);
                R.m[i][j] = 2.0f * ui * uj - (i == j ? 1.0f : 0.0f);
            }
        }
        return R;
    }

    Vec3 axis = vec3_cross(v1, v2);
    float s = vec3_norm(axis);
    Vec3 k = vec3_scale(axis, 1.0f / s);

    // R = I + sin(theta) * K + (1 - cos(theta)) * K^2
    float kx = k.x, ky = k.y, kz = k.z;
    float one_minus_c = 1.0f - c;

    R.m[0][0] = c + kx * kx * one_minus_c;
    R.m[0][1] = kx * ky * one_minus_c - kz * s;
    R.m[0][2] = kx * kz * one_minus_c + ky * s;

    R.m[1][0] = ky * kx * one_minus_c + kz * s;
    R.m[1][1] = c + ky * ky * one_minus_c;
    R.m[1][2] = ky * kz * one_minus_c - kx * s;

    R.m[2][0] = kz * kx * one_minus_c - ky * s;
    R.m[2][1] = kz * ky * one_minus_c + kx * s;
    R.m[2][2] = c + kz * kz * one_minus_c;

    return R;
}

// -------------------------------------------------------------
// Atom & Port Structs
// -------------------------------------------------------------
typedef struct {
    Vec3 pos;             // 3D pseudo-coordinates in Å
    int atomic_number;    // 1=H, 6=C, 7=N, 8=O, 9=F, 14=Si, 16=S, 17=Cl, 35=Br
    float mass;           // in amu
    float charge;         // partial charge e
    float sigma;          // LJ sigma in Å
    float epsilon_k;      // LJ epsilon in Kelvin
    bool is_aromatic;
    bool is_hbd;          // Hydrogen bond donor (-OH, -NH2, -NH-)
    bool is_hba;          // Hydrogen bond acceptor (=O, -O-, -N=, -F)
    int ring_id;          // Ring identifier (0 if acyclic)
} AtomSite;

typedef struct {
    int atom_u;
    int atom_v;
    int bond_order;       // 1 = single, 2 = double, 3 = triple, 4 = aromatic
    bool is_rotatable;
} Bond;

typedef struct {
    int origin_atom;      // Atom index in parent/molecule
    Vec3 pos;             // Port position in Å
    Vec3 normal;          // Outward pointing unit vector
    int state;            // PORT_STATE_EMPTY, OCCUPIED, BLOCKED
    int parent_port_id;
} AttachmentPort;

typedef struct {
    AtomSite atoms[MAX_ATOMS];
    Bond bonds[MAX_BONDS];
    AttachmentPort ports[MAX_PORTS];
    int num_atoms;
    int num_bonds;
    int num_ports;
    int num_attached_fragments;
    float molecular_weight;
    Vec3 center_of_mass;
    float bounding_radius;
} MolecularGraph;

// Computes 1-2 (bonded), 1-3 (angle), and 1-4 (torsion/intra-ring) symmetric exclusion matrix (N x N)
static inline void compute_12_13_exclusions(const MolecularGraph* graph, float* excl_out) {
    int n = graph->num_atoms;
    memset(excl_out, 0, n * n * sizeof(float));
    for (int i = 0; i < n; i++) {
        excl_out[i * n + i] = 1.0f;
    }
    for (int b = 0; b < graph->num_bonds; b++) {
        int u = graph->bonds[b].atom_u;
        int v = graph->bonds[b].atom_v;
        if (u >= 0 && u < n && v >= 0 && v < n) {
            excl_out[u * n + v] = 1.0f;
            excl_out[v * n + u] = 1.0f;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int b1 = 0; b1 < graph->num_bonds; b1++) {
            int w = -1;
            if (graph->bonds[b1].atom_u == i) w = graph->bonds[b1].atom_v;
            else if (graph->bonds[b1].atom_v == i) w = graph->bonds[b1].atom_u;
            if (w < 0 || w >= n) continue;

            for (int b2 = 0; b2 < graph->num_bonds; b2++) {
                int j = -1;
                if (graph->bonds[b2].atom_u == w && graph->bonds[b2].atom_v != i) j = graph->bonds[b2].atom_v;
                else if (graph->bonds[b2].atom_v == w && graph->bonds[b2].atom_u != i) j = graph->bonds[b2].atom_u;
                if (j >= 0 && j < n && j != i) {
                    excl_out[i * n + j] = 1.0f;
                    excl_out[j * n + i] = 1.0f;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int b1 = 0; b1 < graph->num_bonds; b1++) {
            int w = -1;
            if (graph->bonds[b1].atom_u == i) w = graph->bonds[b1].atom_v;
            else if (graph->bonds[b1].atom_v == i) w = graph->bonds[b1].atom_u;
            if (w < 0 || w >= n) continue;

            for (int b2 = 0; b2 < graph->num_bonds; b2++) {
                int k = -1;
                if (graph->bonds[b2].atom_u == w && graph->bonds[b2].atom_v != i) k = graph->bonds[b2].atom_v;
                else if (graph->bonds[b2].atom_v == w && graph->bonds[b2].atom_u != i) k = graph->bonds[b2].atom_u;
                if (k < 0 || k >= n || k == i) continue;

                for (int b3 = 0; b3 < graph->num_bonds; b3++) {
                    int j = -1;
                    if (graph->bonds[b3].atom_u == k && graph->bonds[b3].atom_v != w) j = graph->bonds[b3].atom_v;
                    else if (graph->bonds[b3].atom_v == k && graph->bonds[b3].atom_u != w) j = graph->bonds[b3].atom_u;
                    if (j >= 0 && j < n && j != i && j != w) {
                        excl_out[i * n + j] = 1.0f;
                        excl_out[j * n + i] = 1.0f;
                    }
                }
            }
        }
    }
}

// Fragment definition template for library
typedef struct {
    const char* id;
    const char* name;
    int num_atoms;
    int num_bonds;
    int num_ports;
    bool is_scaffold;
    bool is_cap;
    bool is_linker;
    AtomSite atoms[16];
    Bond bonds[16];
    AttachmentPort ports[4];
} FragmentTemplate;

// -------------------------------------------------------------
// Canonical 3D Local Fragment Library
// -------------------------------------------------------------
static inline FragmentTemplate get_fragment_template(int frag_idx) {
    FragmentTemplate ft;
    memset(&ft, 0, sizeof(FragmentTemplate));

    switch (frag_idx) {
        // --- 0: Benzene Core (Scaffold, 1,4-para attachment) ---
        case 0: {
            ft.id = "benzene_scaffold";
            ft.name = "Benzene 1,4-Core";
            ft.is_scaffold = true;
            ft.num_atoms = 6;
            ft.num_bonds = 6;
            ft.num_ports = 2;

            float r = 1.397f; // C-C aromatic bond length in benzene
            for (int i = 0; i < 6; i++) {
                float angle = i * (float)M_PI / 3.0f;
                ft.atoms[i].pos = vec3_create(r * cosf(angle), r * sinf(angle), 0.0f);
                ft.atoms[i].atomic_number = 6;
                ft.atoms[i].mass = 12.011f;
                ft.atoms[i].charge = -0.115f;
                ft.atoms[i].sigma = 3.40f;
                ft.atoms[i].epsilon_k = 43.27f;
                ft.atoms[i].is_aromatic = true;
                ft.atoms[i].ring_id = 1;
            }
            for (int i = 0; i < 6; i++) {
                ft.bonds[i].atom_u = i;
                ft.bonds[i].atom_v = (i + 1) % 6;
                ft.bonds[i].bond_order = 4;
                ft.bonds[i].is_rotatable = false;
            }
            ft.ports[0] = (AttachmentPort){.origin_atom = 0, .pos = ft.atoms[0].pos, .normal = vec3_create(1.0f, 0.0f, 0.0f), .state = PORT_STATE_EMPTY};
            ft.ports[1] = (AttachmentPort){.origin_atom = 3, .pos = ft.atoms[3].pos, .normal = vec3_create(-1.0f, 0.0f, 0.0f), .state = PORT_STATE_EMPTY};
            break;
        }

        // --- 1: Triphenylamine Core (Scaffold, 3 attachments) ---
        case 1: {
            ft.id = "triphenylamine_core";
            ft.name = "Triphenylamine 3-Arm Core";
            ft.is_scaffold = true;
            ft.num_atoms = 7;
            ft.num_bonds = 6;
            ft.num_ports = 3;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 7;
            ft.atoms[0].mass = 14.007f;
            ft.atoms[0].charge = -0.45f;
            ft.atoms[0].sigma = 3.25f;
            ft.atoms[0].epsilon_k = 85.5f;
            ft.atoms[0].is_hba = true;

            for (int k = 0; k < 3; k++) {
                float angle = k * (2.0f * (float)M_PI / 3.0f);
                Vec3 dir = vec3_create(cosf(angle), sinf(angle), 0.15f * (k % 2 ? 1.0f : -1.0f));
                dir = vec3_normalize(dir);

                int idx1 = 1 + 2 * k;
                int idx2 = 2 + 2 * k;
                ft.atoms[idx1].pos = vec3_scale(dir, 1.42f);
                ft.atoms[idx1].atomic_number = 6;
                ft.atoms[idx1].mass = 12.011f;
                ft.atoms[idx1].sigma = 3.4f;
                ft.atoms[idx1].epsilon_k = 43.3f;
                ft.atoms[idx1].is_aromatic = true;

                ft.atoms[idx2].pos = vec3_scale(dir, 2.82f);
                ft.atoms[idx2].atomic_number = 6;
                ft.atoms[idx2].mass = 12.011f;
                ft.atoms[idx2].sigma = 3.4f;
                ft.atoms[idx2].epsilon_k = 43.3f;
                ft.atoms[idx2].is_aromatic = true;

                ft.bonds[2 * k] = (Bond){.atom_u = 0, .atom_v = idx1, .bond_order = 1, .is_rotatable = true};
                ft.bonds[2 * k + 1] = (Bond){.atom_u = idx1, .atom_v = idx2, .bond_order = 1, .is_rotatable = false};

                ft.ports[k] = (AttachmentPort){.origin_atom = idx2, .pos = ft.atoms[idx2].pos, .normal = dir, .state = PORT_STATE_EMPTY};
            }
            break;
        }

        // --- 2: Adamantane Rigid Scaffold (2 attachments) ---
        case 2: {
            ft.id = "adamantane_core";
            ft.name = "1,3-Adamantane Core";
            ft.is_scaffold = true;
            ft.num_atoms = 10;
            ft.num_bonds = 12;
            ft.num_ports = 2;

            float d = 0.889f;
            float m = 1.257f;
            Vec3 cage[10] = {
                { d,  d,  d}, // 0: Bridgehead A0
                { d, -d, -d}, // 1: Bridgehead A1
                {-d,  d, -d}, // 2: Bridgehead A2
                {-d, -d,  d}, // 3: Bridgehead A3
                { m, 0.0f, 0.0f}, // 4: Methylene M01 (between A0, A1)
                {-m, 0.0f, 0.0f}, // 5: Methylene M23 (between A2, A3)
                {0.0f,  m, 0.0f}, // 6: Methylene M02 (between A0, A2)
                {0.0f, -m, 0.0f}, // 7: Methylene M13 (between A1, A3)
                {0.0f, 0.0f,  m}, // 8: Methylene M03 (between A0, A3)
                {0.0f, 0.0f, -m}  // 9: Methylene M12 (between A1, A2)
            };
            for (int i = 0; i < 10; i++) {
                ft.atoms[i].pos = cage[i];
                ft.atoms[i].atomic_number = 6;
                ft.atoms[i].mass = 12.011f;
                ft.atoms[i].sigma = 3.50f;
                ft.atoms[i].epsilon_k = 33.2f;
                ft.atoms[i].ring_id = 2;
            }
            int bond_pairs[12][2] = {
                {0, 4}, {0, 6}, {0, 8}, // A0 to M01, M02, M03
                {1, 4}, {1, 7}, {1, 9}, // A1 to M01, M13, M12
                {2, 5}, {2, 6}, {2, 9}, // A2 to M23, M02, M12
                {3, 5}, {3, 7}, {3, 8}  // A3 to M23, M13, M03
            };
            for (int b = 0; b < 12; b++) {
                ft.bonds[b] = (Bond){.atom_u = bond_pairs[b][0], .atom_v = bond_pairs[b][1], .bond_order = 1, .is_rotatable = false};
            }
            ft.ports[0] = (AttachmentPort){.origin_atom = 0, .pos = ft.atoms[0].pos, .normal = vec3_create(0.577f, 0.577f, 0.577f), .state = PORT_STATE_EMPTY};
            ft.ports[1] = (AttachmentPort){.origin_atom = 1, .pos = ft.atoms[1].pos, .normal = vec3_create(0.577f, -0.577f, -0.577f), .state = PORT_STATE_EMPTY};
            break;
        }

        // --- 3: Para-Phenylene Linker (Rigid aromatic I-beam) ---
        case 3: {
            ft.id = "para_phenylene";
            ft.name = "Para-Phenylene Linker";
            ft.is_linker = true;
            ft.num_atoms = 6;
            ft.num_bonds = 6;
            ft.num_ports = 1;

            float r = 1.397f;
            for (int i = 0; i < 6; i++) {
                float angle = i * (float)M_PI / 3.0f;
                ft.atoms[i].pos = vec3_create(r * cosf(angle), r * sinf(angle), 0.0f);
                ft.atoms[i].atomic_number = 6;
                ft.atoms[i].mass = 12.011f;
                ft.atoms[i].sigma = 3.40f;
                ft.atoms[i].epsilon_k = 43.27f;
                ft.atoms[i].is_aromatic = true;
            }
            for (int i = 0; i < 6; i++) {
                ft.bonds[i].atom_u = i;
                ft.bonds[i].atom_v = (i + 1) % 6;
                ft.bonds[i].bond_order = 4;
                ft.bonds[i].is_rotatable = false;
            }
            ft.ports[0] = (AttachmentPort){.origin_atom = 3, .pos = ft.atoms[3].pos, .normal = vec3_create(-1.0f, 0.0f, 0.0f), .state = PORT_STATE_EMPTY};
            break;
        }

        // --- 4: Ethylene / Aliphatic Chain Linker (Flexible, Springy, Rotatable) ---
        case 4: {
            ft.id = "ethylene_linker";
            ft.name = "Ethylene Di-Carbon Linker";
            ft.is_linker = true;
            ft.num_atoms = 2;
            ft.num_bonds = 1;
            ft.num_ports = 1;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 6;
            ft.atoms[0].mass = 12.011f;
            ft.atoms[0].sigma = 3.50f;
            ft.atoms[0].epsilon_k = 33.2f;

            ft.atoms[1].pos = vec3_create(1.54f, 0.45f, 0.0f);
            ft.atoms[1].atomic_number = 6;
            ft.atoms[1].mass = 12.011f;
            ft.atoms[1].sigma = 3.50f;
            ft.atoms[1].epsilon_k = 33.2f;

            ft.bonds[0] = (Bond){.atom_u = 0, .atom_v = 1, .bond_order = 1, .is_rotatable = true};
            ft.ports[0] = (AttachmentPort){.origin_atom = 1, .pos = ft.atoms[1].pos, .normal = vec3_create(0.9f, 0.43f, 0.0f), .state = PORT_STATE_EMPTY};
            break;
        }

        // --- 5: Thiophene Bridge Linker (Conjugated Heterocycle) ---
        case 5: {
            ft.id = "thiophene_bridge";
            ft.name = "Thiophene Bridge Linker";
            ft.is_linker = true;
            ft.num_atoms = 5;
            ft.num_bonds = 5;
            ft.num_ports = 1;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[1].pos = vec3_create(0.72f, 1.25f, 0.0f);
            ft.atoms[2].pos = vec3_create(2.10f, 0.85f, 0.0f);
            ft.atoms[3].pos = vec3_create(1.95f, -0.45f, 0.0f);
            ft.atoms[4].pos = vec3_create(0.75f, -0.85f, 0.0f);

            for (int i = 0; i < 5; i++) {
                ft.atoms[i].atomic_number = (i == 1) ? 16 : 6;
                ft.atoms[i].mass = (i == 1) ? 32.06f : 12.011f;
                ft.atoms[i].sigma = (i == 1) ? 3.55f : 3.40f;
                ft.atoms[i].epsilon_k = (i == 1) ? 125.0f : 43.3f;
                ft.atoms[i].is_aromatic = true;
                ft.atoms[i].ring_id = 3;
            }
            for (int i = 0; i < 5; i++) {
                ft.bonds[i] = (Bond){.atom_u = i, .atom_v = (i + 1) % 5, .bond_order = 4, .is_rotatable = false};
            }
            ft.ports[0] = (AttachmentPort){.origin_atom = 2, .pos = ft.atoms[2].pos, .normal = vec3_create(0.85f, 0.52f, 0.0f), .state = PORT_STATE_EMPTY};
            break;
        }

        // --- 6: Hydrogen Cap (Terminal H) ---
        case 6: {
            ft.id = "hydrogen_cap";
            ft.name = "Hydrogen Cap";
            ft.is_cap = true;
            ft.num_atoms = 1;
            ft.num_bonds = 0;
            ft.num_ports = 0;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 1;
            ft.atoms[0].mass = 1.008f;
            ft.atoms[0].charge = 0.115f;
            ft.atoms[0].sigma = 2.42f;
            ft.atoms[0].epsilon_k = 7.55f;
            break;
        }

        // --- 7: Amine Functional Cap (-NH2, Strong H-Bond Donor) ---
        case 7: {
            ft.id = "amine_cap";
            ft.name = "Amine Cap (-NH2)";
            ft.is_cap = true;
            ft.num_atoms = 3;
            ft.num_bonds = 2;
            ft.num_ports = 0;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 7;
            ft.atoms[0].mass = 14.007f;
            ft.atoms[0].charge = -0.65f;
            ft.atoms[0].sigma = 3.25f;
            ft.atoms[0].epsilon_k = 85.5f;
            ft.atoms[0].is_hbd = true;
            ft.atoms[0].is_hba = true;

            ft.atoms[1].pos = vec3_create(0.82f, 0.58f, 0.0f);
            ft.atoms[1].atomic_number = 1;
            ft.atoms[1].mass = 1.008f;
            ft.atoms[1].charge = 0.325f;
            ft.atoms[1].sigma = 1.07f;
            ft.atoms[1].epsilon_k = 7.9f;

            ft.atoms[2].pos = vec3_create(0.82f, -0.58f, 0.0f);
            ft.atoms[2].atomic_number = 1;
            ft.atoms[2].mass = 1.008f;
            ft.atoms[2].charge = 0.325f;
            ft.atoms[2].sigma = 1.07f;
            ft.atoms[2].epsilon_k = 7.9f;

            ft.bonds[0] = (Bond){.atom_u = 0, .atom_v = 1, .bond_order = 1, .is_rotatable = false};
            ft.bonds[1] = (Bond){.atom_u = 0, .atom_v = 2, .bond_order = 1, .is_rotatable = false};
            break;
        }

        // --- 8: Hydroxyl Cap (-OH, Sacrificial H-Bond Donor / Toughness) ---
        case 8: {
            ft.id = "hydroxyl_cap";
            ft.name = "Hydroxyl Cap (-OH)";
            ft.is_cap = true;
            ft.num_atoms = 2;
            ft.num_bonds = 1;
            ft.num_ports = 0;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 8;
            ft.atoms[0].mass = 15.999f;
            ft.atoms[0].charge = -0.58f;
            ft.atoms[0].sigma = 3.06f;
            ft.atoms[0].epsilon_k = 105.8f;
            ft.atoms[0].is_hbd = true;
            ft.atoms[0].is_hba = true;

            ft.atoms[1].pos = vec3_create(0.78f, 0.55f, 0.0f);
            ft.atoms[1].atomic_number = 1;
            ft.atoms[1].mass = 1.008f;
            ft.atoms[1].charge = 0.58f;
            ft.atoms[1].sigma = 1.06f;
            ft.atoms[1].epsilon_k = 7.55f;

            ft.bonds[0] = (Bond){.atom_u = 0, .atom_v = 1, .bond_order = 1, .is_rotatable = true};
            break;
        }

        // --- 9: Bulky Tert-Butyl Cap (-C(CH3)3, High Steric Occlusion) ---
        case 9: {
            ft.id = "tert_butyl_cap";
            ft.name = "Tert-Butyl Cap";
            ft.is_cap = true;
            ft.num_atoms = 4;
            ft.num_bonds = 3;
            ft.num_ports = 0;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 6;
            ft.atoms[0].mass = 12.011f;
            ft.atoms[0].sigma = 3.50f;
            ft.atoms[0].epsilon_k = 33.2f;

            ft.atoms[1].pos = vec3_create(1.25f, 0.88f, 0.0f);
            ft.atoms[2].pos = vec3_create(1.25f, -0.44f, 0.76f);
            ft.atoms[3].pos = vec3_create(1.25f, -0.44f, -0.76f);

            for (int k = 1; k <= 3; k++) {
                ft.atoms[k].atomic_number = 6;
                ft.atoms[k].mass = 12.011f;
                ft.atoms[k].sigma = 3.50f;
                ft.atoms[k].epsilon_k = 33.2f;
                ft.bonds[k - 1] = (Bond){.atom_u = 0, .atom_v = k, .bond_order = 1, .is_rotatable = true};
            }
            break;
        }

        // --- 10: Trifluoromethyl Cap (-CF3, High Electron Affinity & Rigidity) ---
        case 10: {
            ft.id = "trifluoromethyl_cap";
            ft.name = "Trifluoromethyl Cap (-CF3)";
            ft.is_cap = true;
            ft.num_atoms = 4;
            ft.num_bonds = 3;
            ft.num_ports = 0;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[0].atomic_number = 6;
            ft.atoms[0].mass = 12.011f;
            ft.atoms[0].charge = 0.54f;
            ft.atoms[0].sigma = 3.50f;
            ft.atoms[0].epsilon_k = 33.2f;

            ft.atoms[1].pos = vec3_create(1.15f, 0.78f, 0.0f);
            ft.atoms[2].pos = vec3_create(1.15f, -0.39f, 0.67f);
            ft.atoms[3].pos = vec3_create(1.15f, -0.39f, -0.67f);

            for (int k = 1; k <= 3; k++) {
                ft.atoms[k].atomic_number = 9;
                ft.atoms[k].mass = 18.998f;
                ft.atoms[k].charge = -0.18f;
                ft.atoms[k].sigma = 3.12f;
                ft.atoms[k].epsilon_k = 30.7f;
                ft.atoms[k].is_hba = true;
                ft.bonds[k - 1] = (Bond){.atom_u = 0, .atom_v = k, .bond_order = 1, .is_rotatable = false};
            }
            break;
        }

        // --- 11: Cyanovinyl Cap (-C=C-CN, High Polar Surface Area & Stiff Dipole) ---
        case 11: {
            ft.id = "cyanovinyl_cap";
            ft.name = "Cyanovinyl Cap (-C=C-CN)";
            ft.is_cap = true;
            ft.num_atoms = 4;
            ft.num_bonds = 3;
            ft.num_ports = 0;

            ft.atoms[0].pos = vec3_create(0.0f, 0.0f, 0.0f);
            ft.atoms[1].pos = vec3_create(1.34f, 0.0f, 0.0f);
            ft.atoms[2].pos = vec3_create(2.78f, 0.0f, 0.0f);
            ft.atoms[3].pos = vec3_create(3.93f, 0.0f, 0.0f);

            for (int k = 0; k < 3; k++) {
                ft.atoms[k].atomic_number = 6;
                ft.atoms[k].mass = 12.011f;
                ft.atoms[k].sigma = 3.40f;
                ft.atoms[k].epsilon_k = 43.3f;
            }
            ft.atoms[3].atomic_number = 7;
            ft.atoms[3].mass = 14.007f;
            ft.atoms[3].charge = -0.55f;
            ft.atoms[3].sigma = 3.25f;
            ft.atoms[3].epsilon_k = 85.5f;
            ft.atoms[3].is_hba = true;

            ft.bonds[0] = (Bond){.atom_u = 0, .atom_v = 1, .bond_order = 2, .is_rotatable = false};
            ft.bonds[1] = (Bond){.atom_u = 1, .atom_v = 2, .bond_order = 1, .is_rotatable = false};
            ft.bonds[2] = (Bond){.atom_u = 2, .atom_v = 3, .bond_order = 3, .is_rotatable = false};
            break;
        }

        default:
            return get_fragment_template(6);
    }

    return ft;
}

// -------------------------------------------------------------
// SE(3) Rigid-Body Attachment Execution
// -------------------------------------------------------------
static inline bool rigid_body_attach(MolecularGraph* graph, int port_idx, int frag_idx) {
    if (port_idx < 0 || port_idx >= graph->num_ports) return false;
    AttachmentPort* p_a = &graph->ports[port_idx];
    if (p_a->state != PORT_STATE_EMPTY) return false;

    FragmentTemplate ft = get_fragment_template(frag_idx);
    if (graph->num_atoms + ft.num_atoms > MAX_ATOMS) return false;
    if (graph->num_bonds + ft.num_bonds + 1 > MAX_BONDS) return false;

    Vec3 p_b = ft.atoms[0].pos;
    Vec3 c_frag = vec3_create(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < ft.num_atoms; k++) {
        c_frag = vec3_add(c_frag, ft.atoms[k].pos);
    }
    c_frag = vec3_scale(c_frag, 1.0f / (float)ft.num_atoms);

    Vec3 delta_c = vec3_sub(p_b, c_frag);
    Vec3 u_b = (vec3_norm(delta_c) > 1e-3f) ? vec3_normalize(delta_c) : vec3_create(-1.0f, 0.0f, 0.0f);
    Vec3 target_u = vec3_scale(p_a->normal, -1.0f);
    Mat3 R = mat3_align_vectors(u_b, target_u);

    float d_bond = 1.48f;
    Vec3 bond_disp = vec3_scale(p_a->normal, d_bond);
    Vec3 target_pos = vec3_add(p_a->pos, bond_disp);
    Vec3 rotated_pb = mat3_vec3_mul(R, p_b);
    Vec3 trans = vec3_sub(target_pos, rotated_pb);

    Vec3 transformed_pos[16];
    for (int i = 0; i < ft.num_atoms; i++) {
        Vec3 rot_pos = mat3_vec3_mul(R, ft.atoms[i].pos);
        transformed_pos[i] = vec3_add(rot_pos, trans);

        for (int j = 0; j < graph->num_atoms; j++) {
            if (j == p_a->origin_atom) continue;
            bool is_direct_nbr = false;
            for (int b = 0; b < graph->num_bonds; b++) {
                if ((graph->bonds[b].atom_u == p_a->origin_atom && graph->bonds[b].atom_v == j) ||
                    (graph->bonds[b].atom_v == p_a->origin_atom && graph->bonds[b].atom_u == j)) {
                    is_direct_nbr = true;
                    break;
                }
            }
            if (is_direct_nbr) continue;

            float dist = vec3_dist(transformed_pos[i], graph->atoms[j].pos);
            float min_dist = 0.65f * 0.5f * (ft.atoms[i].sigma + graph->atoms[j].sigma);
            if (dist < min_dist) {
                return false;
            }
        }
    }

    int atom_offset = graph->num_atoms;
    for (int i = 0; i < ft.num_atoms; i++) {
        AtomSite site = ft.atoms[i];
        site.pos = transformed_pos[i];
        graph->atoms[atom_offset + i] = site;
    }
    graph->num_atoms += ft.num_atoms;

    int bond_idx = graph->num_bonds;
    graph->bonds[bond_idx] = (Bond){
        .atom_u = p_a->origin_atom,
        .atom_v = atom_offset,
        .bond_order = 1,
        .is_rotatable = true
    };
    graph->num_bonds++;

    for (int b = 0; b < ft.num_bonds; b++) {
        Bond nb = ft.bonds[b];
        nb.atom_u += atom_offset;
        nb.atom_v += atom_offset;
        graph->bonds[graph->num_bonds++] = nb;
    }

    p_a->state = PORT_STATE_OCCUPIED;

    for (int p = 0; p < ft.num_ports; p++) {
        if (graph->num_ports < MAX_PORTS) {
            AttachmentPort np = ft.ports[p];
            Vec3 rot_port_pos = mat3_vec3_mul(R, np.pos);
            np.pos = vec3_add(rot_port_pos, trans);
            np.normal = vec3_normalize(mat3_vec3_mul(R, np.normal));
            np.origin_atom += atom_offset;
            np.state = PORT_STATE_EMPTY;
            np.parent_port_id = port_idx;
            graph->ports[graph->num_ports++] = np;
        }
    }

    graph->num_attached_fragments++;
    return true;
}

// -------------------------------------------------------------
// Initialize Base Scaffold Molecule
// -------------------------------------------------------------
static inline void init_base_scaffold(MolecularGraph* graph, int scaffold_frag_idx) {
    memset(graph, 0, sizeof(MolecularGraph));
    FragmentTemplate ft = get_fragment_template(scaffold_frag_idx);

    graph->num_atoms = ft.num_atoms;
    graph->num_bonds = ft.num_bonds;
    graph->num_ports = ft.num_ports;
    graph->num_attached_fragments = 0;

    for (int i = 0; i < ft.num_atoms; i++) {
        graph->atoms[i] = ft.atoms[i];
    }
    for (int b = 0; b < ft.num_bonds; b++) {
        graph->bonds[b] = ft.bonds[b];
    }
    for (int p = 0; p < ft.num_ports; p++) {
        graph->ports[p] = ft.ports[p];
    }
}

#endif // MOL_GRAPH_H
