#include "cdft_swarm.h"
#include <time.h>

int main(int argc, char** argv) {
    bool interactive = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) {
            interactive = true;
        }
    }

    printf("=================================================================\n");
    printf("  dens-city: Stage 1 Multi-Objective RL Swarm Standalone Runner\n");
    printf("=================================================================\n");

    Env env;
    memset(&env, 0, sizeof(Env));
    env.num_agents = 1;
    env.rng = (unsigned int)time(NULL);

    env.observations = (float*)calloc(TOTAL_OBS_SIZE, sizeof(float));
    env.actions = (float*)calloc(2, sizeof(float));
    env.rewards = (float*)calloc(1, sizeof(float));
    env.terminals = (unsigned char*)calloc(1, sizeof(unsigned char));
    env.action_mask = (unsigned char*)calloc(TOTAL_ACTION_MASK_SIZE, sizeof(unsigned char));

    c_reset(&env);

    if (interactive) {
        printf("[*] Running in Interactive 3D Raylib Mode. Press ESC to exit.\n");
        c_render(&env);

        int tick = 0;
        while (!WindowShouldClose()) {
            tick++;
            if (tick % 30 == 0) { // Step every 0.5s for clear visual inspection
                // Sample valid port from action mask
                int valid_ports[MAX_PORTS];
                int n_valid_ports = 0;
                for (int p = 0; p < MAX_PORTS; p++) {
                    if (env.action_mask[p]) valid_ports[n_valid_ports++] = p;
                }

                // Sample valid fragment from action mask
                int valid_frags[NUM_FRAGMENT_CHOICES + 1];
                int n_valid_frags = 0;
                unsigned char* frag_mask = &env.action_mask[MAX_PORTS];
                for (int f = 0; f <= NUM_FRAGMENT_CHOICES; f++) {
                    if (frag_mask[f]) valid_frags[n_valid_frags++] = f;
                }

                if (n_valid_ports > 0 && n_valid_frags > 0) {
                    env.actions[0] = (float)valid_ports[rand_r(&env.rng) % n_valid_ports];
                    env.actions[1] = (float)valid_frags[rand_r(&env.rng) % n_valid_frags];
                } else {
                    env.actions[0] = 0;
                    env.actions[1] = ACTION_FINALIZE;
                }

                c_step(&env);
            }
            c_render(&env);
        }
        c_close(&env);
    } else {
        printf("[*] Running headless benchmark (10,000 steps)...\n");
        clock_t t0 = clock();

        int num_episodes = 0;
        int total_steps = 10000;
        for (int step = 0; step < total_steps; step++) {
            int valid_ports[MAX_PORTS];
            int n_valid_ports = 0;
            for (int p = 0; p < MAX_PORTS; p++) {
                if (env.action_mask[p]) valid_ports[n_valid_ports++] = p;
            }

            int valid_frags[NUM_FRAGMENT_CHOICES + 1];
            int n_valid_frags = 0;
            unsigned char* frag_mask = &env.action_mask[MAX_PORTS];
            for (int f = 0; f <= NUM_FRAGMENT_CHOICES; f++) {
                if (frag_mask[f]) valid_frags[n_valid_frags++] = f;
            }

            if (n_valid_ports > 0 && n_valid_frags > 0) {
                env.actions[0] = (float)valid_ports[rand_r(&env.rng) % n_valid_ports];
                env.actions[1] = (float)valid_frags[rand_r(&env.rng) % n_valid_frags];
            } else {
                env.actions[0] = 0;
                env.actions[1] = ACTION_FINALIZE;
            }

            c_step(&env);
            if (env.terminals[0]) {
                num_episodes++;
            }
        }

        clock_t t1 = clock();
        double elapsed_sec = (double)(t1 - t0) / CLOCKS_PER_SEC;
        double steps_per_sec = (double)total_steps / elapsed_sec;

        printf("[+] Completed %d steps across %d episodes in %.3f s\n", total_steps, num_episodes, elapsed_sec);
        printf("[+] Throughput: %.1f steps/sec (Single CPU Core)\n", steps_per_sec);
        printf("[+] Telemetry Summary:\n");
        printf("    -> Valid Molecules Generated: %.0f\n", env.log.valid_molecules);
        printf("    -> Avg Episode Return: %.2f\n", (env.log.n > 0) ? (env.log.score / env.log.n) : 0.0f);
        printf("    -> Last P_wall: %.2f bar | Solvation Energy: %.2f kcal/mol\n", env.log.p_wall, env.log.omega_solv);
    }

    free(env.observations);
    free(env.actions);
    free(env.rewards);
    free(env.terminals);
    free(env.action_mask);

    return 0;
}
