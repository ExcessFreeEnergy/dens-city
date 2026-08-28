#ifndef MECHANICS_EVAL_H
#define MECHANICS_EVAL_H

#include "mol_graph.h"
#include <math.h>
#include <string.h>

// Struct holding the computed mechanical and topological proxy scores
typedef struct {
    // 1. Elasticity Matrix (Springy / Bouncy)
    float rotatable_bond_fraction;
    float kuhn_persistence_proxy;   // Ratio of 3D end-to-end span to contour length (low = coiled/springy, high = stiff)
    float crosslink_node_distance;  // 3D distance between terminal ports in Å
    float steric_hindrance_penalty;

    // 2. Tensile Strength Matrix (High Yield Stress)
    float aromatic_density;         // Aromatic heavy atoms / total heavy atoms
    float pmi_linearity;            // Rod-like score from inertia tensor eigenvalues (1.0 = perfect rod)
    float pmi_planar;               // Disc-like score (1.0 = planar disc)
    int multivalency_count;         // Total active / reactive crosslinking ports

    // 3. Fracture Toughness Matrix (Cracking Resistance)
    int hbd_count;                  // Hydrogen bond donors (-OH, -NH2, -NH-)
    int hba_count;                  // Hydrogen bond acceptors (=O, -O-, -N=)
    float sacrificial_hbond_score;  // Close-proximity H-bond pairs that can dissipate shock
    float pi_pi_stacking_score;     // Proximity of distinct aromatic rings
    float conformational_entropy;   // Estimated accessible rotamer degrees of freedom

    // 4. Weight & Density Matrix (Ultra-Lightweight)
    float fractional_free_volume;   // Ratio of bounding ellipsoid volume to vdW volume
    float heavy_atom_penalty;       // Penalty for halogens / heavy metalloids
    float tpsa_proxy;               // Topological Polar Surface Area (Å^2)
    float molecular_weight;         // Total mass in amu
} MechanicsProfile;

// Jacobi eigenvalue solver for symmetric 3x3 inertia matrix
static inline void diagonalize_symmetric_3x3(float A[3][3], float eigvals[3], float eigvecs[3][3]) {
    // Initialize eigvecs to identity
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            eigvecs[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    float D[3][3];
    memcpy(D, A, sizeof(D));

    // 5 Jacobi iterations
    for (int iter = 0; iter < 10; iter++) {
        // Find largest off-diagonal element
        int p = 0, q = 1;
        float max_off = fabsf(D[0][1]);
        if (fabsf(D[0][2]) > max_off) { max_off = fabsf(D[0][2]); p = 0; q = 2; }
        if (fabsf(D[1][2]) > max_off) { max_off = fabsf(D[1][2]); p = 1; q = 2; }

        if (max_off < 1e-6f) break;

        float app = D[p][p];
        float aqq = D[q][q];
        float apq = D[p][q];

        float theta = 0.5f * atan2f(2.0f * apq, aqq - app);
        float c = cosf(theta);
        float s = sinf(theta);

        // Update D matrix
        D[p][p] = c * c * app - 2.0f * s * c * apq + s * s * aqq;
        D[q][q] = s * s * app + 2.0f * s * c * apq + c * c * aqq;
        D[p][q] = 0.0f;
        D[q][p] = 0.0f;

        int r = 3 - p - q; // the third index
        float apr = D[p][r];
        float aqr = D[q][r];
        D[p][r] = c * apr - s * aqr;
        D[r][p] = D[p][r];
        D[q][r] = s * apr + c * aqr;
        D[r][q] = D[q][r];

        // Update eigenvectors
        for (int i = 0; i < 3; i++) {
            float vip = eigvecs[i][p];
            float viq = eigvecs[i][q];
            eigvecs[i][p] = c * vip - s * viq;
            eigvecs[i][q] = s * vip + c * viq;
        }
    }

    eigvals[0] = D[0][0];
    eigvals[1] = D[1][1];
    eigvals[2] = D[2][2];

    // Sort eigenvalues ascending: eigvals[0] <= eigvals[1] <= eigvals[2]
    for (int i = 0; i < 2; i++) {
        for (int j = i + 1; j < 3; j++) {
            if (eigvals[j] < eigvals[i]) {
                float tmp = eigvals[i];
                eigvals[i] = eigvals[j];
                eigvals[j] = tmp;

                for (int k = 0; k < 3; k++) {
                    float vtmp = eigvecs[k][i];
                    eigvecs[k][i] = eigvecs[k][j];
                    eigvecs[k][j] = vtmp;
                }
            }
        }
    }
}

// -------------------------------------------------------------
// Evaluate Microscopic Mechanics Profile in Sub-Microsecond
// -------------------------------------------------------------
static inline MechanicsProfile evaluate_mechanics(const MolecularGraph* graph) {
    MechanicsProfile prof;
    memset(&prof, 0, sizeof(MechanicsProfile));

    int n_atoms = graph->num_atoms;
    if (n_atoms == 0) return prof;

    // 1. Center of Mass & Total Mass
    float total_mass = 0.0f;
    Vec3 com = {0.0f, 0.0f, 0.0f};
    int heavy_atoms = 0;
    int aromatic_heavy = 0;
    int rotatable_bonds = 0;
    float vdw_volume = 0.0f;

    for (int i = 0; i < n_atoms; i++) {
        const AtomSite* a = &graph->atoms[i];
        total_mass += a->mass;
        com = vec3_add(com, vec3_scale(a->pos, a->mass));

        if (a->atomic_number > 1) {
            heavy_atoms++;
            if (a->is_aromatic) aromatic_heavy++;
        }

        // TPSA proxy: O, N, S, polar H contributions
        if (a->atomic_number == 8) prof.tpsa_proxy += (a->is_hbd ? 20.23f : 9.23f);
        else if (a->atomic_number == 7) prof.tpsa_proxy += (a->is_hbd ? 26.02f : 12.89f);
        else if (a->atomic_number == 16) prof.tpsa_proxy += 25.3f;

        if (a->is_hbd) prof.hbd_count++;
        if (a->is_hba) prof.hba_count++;

        // Heavy atom penalties (halogens F, Cl, Br or heavy metalloids)
        if (a->atomic_number == 9) prof.heavy_atom_penalty += 0.5f;
        else if (a->atomic_number == 17) prof.heavy_atom_penalty += 2.0f;
        else if (a->atomic_number == 35) prof.heavy_atom_penalty += 5.0f;
        else if (a->atomic_number > 16) prof.heavy_atom_penalty += 1.5f;

        // VdW Volume: 4/3 * pi * (sigma/2)^3
        float r_vdw = 0.5f * a->sigma;
        vdw_volume += (4.0f / 3.0f) * (float)M_PI * (r_vdw * r_vdw * r_vdw);
    }

    if (total_mass > 0.0f) {
        com = vec3_scale(com, 1.0f / total_mass);
    }
    prof.molecular_weight = total_mass;

    // Rotatable bonds
    for (int b = 0; b < graph->num_bonds; b++) {
        if (graph->bonds[b].is_rotatable) {
            rotatable_bonds++;
        }
    }
    prof.rotatable_bond_fraction = (graph->num_bonds > 0) ? ((float)rotatable_bonds / (float)graph->num_bonds) : 0.0f;
    prof.aromatic_density = (heavy_atoms > 0) ? ((float)aromatic_heavy / (float)heavy_atoms) : 0.0f;

    // 2. Principal Moments of Inertia (PMI)
    float I_tensor[3][3] = {{0}};
    for (int i = 0; i < n_atoms; i++) {
        const AtomSite* a = &graph->atoms[i];
        Vec3 r = vec3_sub(a->pos, com);
        float m = a->mass;

        I_tensor[0][0] += m * (r.y * r.y + r.z * r.z);
        I_tensor[1][1] += m * (r.x * r.x + r.z * r.z);
        I_tensor[2][2] += m * (r.x * r.x + r.y * r.y);

        I_tensor[0][1] -= m * r.x * r.y;
        I_tensor[1][0] -= m * r.x * r.y;

        I_tensor[0][2] -= m * r.x * r.z;
        I_tensor[2][0] -= m * r.x * r.z;

        I_tensor[1][2] -= m * r.y * r.z;
        I_tensor[2][1] -= m * r.y * r.z;
    }

    float eigvals[3];
    float eigvecs[3][3];
    diagonalize_symmetric_3x3(I_tensor, eigvals, eigvecs);
    // Principal moments I1 <= I2 <= I3
    float I1 = fmaxf(1e-3f, eigvals[0]);
    float I2 = fmaxf(1e-3f, eigvals[1]);
    float I3 = fmaxf(1e-3f, eigvals[2]);

    // Linearity (rod-like): for a long cylinder along principal axis 1, I1 << I2 ≈ I3 -> (I3 - I1) / I3 ≈ 1 and (I3 - I2) / I3 ≈ 0
    prof.pmi_linearity = fmaxf(0.0f, (I3 - I2) / I3); // standard PMI shape index: (I3 - I2) / (I3 - I1) or rod ratio (I2 - I1) / I3
    // Alternate standard rod linearity score: 1 - (I1 / I3)
    if (I3 > 1e-3f) {
        prof.pmi_linearity = 1.0f - (I1 / I3);
        prof.pmi_planar = (2.0f * (I2 - I1)) / I3;
    }

    // 3. 3D Spatial Bounding Span & Kuhn / Persistence Length Proxy
    float max_span = 0.0f;
    for (int i = 0; i < n_atoms; i++) {
        for (int j = i + 1; j < n_atoms; j++) {
            float d = vec3_dist(graph->atoms[i].pos, graph->atoms[j].pos);
            if (d > max_span) max_span = d;
        }
    }
    float contour_length = fmaxf(1.0f, (float)graph->num_bonds * 1.48f);
    // Ratio of maximum end-to-end span to contour length: coiled spring has low ratio (< 0.5), stiff rod has high ratio (~ 1.0)
    prof.kuhn_persistence_proxy = max_span / contour_length;

    // Crosslink node distance (between active/terminal ports)
    int empty_ports = 0;
    float total_port_dist = 0.0f;
    int port_pairs = 0;
    for (int p1 = 0; p1 < graph->num_ports; p1++) {
        if (graph->ports[p1].state == PORT_STATE_EMPTY) {
            empty_ports++;
            for (int p2 = p1 + 1; p2 < graph->num_ports; p2++) {
                if (graph->ports[p2].state == PORT_STATE_EMPTY) {
                    total_port_dist += vec3_dist(graph->ports[p1].pos, graph->ports[p2].pos);
                    port_pairs++;
                }
            }
        }
    }
    prof.multivalency_count = empty_ports;
    prof.crosslink_node_distance = (port_pairs > 0) ? (total_port_dist / (float)port_pairs) : max_span;

    // 4. Sacrificial Hydrogen Bonding & π-π Stacking Interactions
    for (int i = 0; i < n_atoms; i++) {
        if (graph->atoms[i].is_hbd) {
            for (int j = 0; j < n_atoms; j++) {
                if (i != j && graph->atoms[j].is_hba) {
                    float d = vec3_dist(graph->atoms[i].pos, graph->atoms[j].pos);
                    if (d >= 2.2f && d <= 3.8f) {
                        prof.sacrificial_hbond_score += 1.0f;
                    }
                }
            }
        }
    }

    // Aromatic ring centroid π-π stacking score
    Vec3 ring_centroids[8];
    int ring_counts[8] = {0};
    memset(ring_centroids, 0, sizeof(ring_centroids));
    for (int i = 0; i < n_atoms; i++) {
        int r_id = graph->atoms[i].ring_id;
        if (r_id > 0 && r_id < 8 && graph->atoms[i].is_aromatic) {
            ring_centroids[r_id] = vec3_add(ring_centroids[r_id], graph->atoms[i].pos);
            ring_counts[r_id]++;
        }
    }
    for (int r = 1; r < 8; r++) {
        if (ring_counts[r] > 0) {
            ring_centroids[r] = vec3_scale(ring_centroids[r], 1.0f / (float)ring_counts[r]);
        }
    }
    for (int r1 = 1; r1 < 8; r1++) {
        if (ring_counts[r1] == 0) continue;
        for (int r2 = r1 + 1; r2 < 8; r2++) {
            if (ring_counts[r2] == 0) continue;
            float r_dist = vec3_dist(ring_centroids[r1], ring_centroids[r2]);
            if (r_dist >= 3.2f && r_dist <= 5.8f) {
                prof.pi_pi_stacking_score += 1.0f;
            }
        }
    }

    // 5. Conformational Entropy Proxy
    prof.conformational_entropy = (float)rotatable_bonds * 1.10f; // ~ ln(3) ≈ 1.10 per rotamer dihedral

    // 6. Fractional Free Volume (FFV) Proxy
    // Approximate bounding ellipsoid semi-axes from principal moments of inertia: I_x = m/5 * (b^2 + c^2), etc.
    float a_ellip = sqrtf(fmaxf(1.0f, (5.0f / (2.0f * fmaxf(1.0f, total_mass))) * (I2 + I3 - I1)));
    float b_ellip = sqrtf(fmaxf(1.0f, (5.0f / (2.0f * fmaxf(1.0f, total_mass))) * (I1 + I3 - I2)));
    float c_ellip = sqrtf(fmaxf(1.0f, (5.0f / (2.0f * fmaxf(1.0f, total_mass))) * (I1 + I2 - I3)));
    float ellip_volume = (4.0f / 3.0f) * (float)M_PI * a_ellip * b_ellip * c_ellip;

    prof.fractional_free_volume = (ellip_volume > vdw_volume) ? ((ellip_volume - vdw_volume) / ellip_volume) : 0.1f;

    return prof;
}

#endif // MECHANICS_EVAL_H
