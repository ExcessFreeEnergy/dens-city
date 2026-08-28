#include "cdft_swarm.h"

#define OBS_SIZE TOTAL_OBS_SIZE
#define NUM_ATNS 2
#define ACT_SIZES {MAX_PORTS, NUM_FRAGMENT_CHOICES + 1}
#define OBS_TENSOR_T FloatTensor
#define MY_ACTION_MASK
#define ACTION_MASK_SIZE TOTAL_ACTION_MASK_SIZE

#define Env CDFT_Swarm_Env
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    if (kwargs) {
        DictItem* item = dict_get_unsafe(kwargs, "target_elasticity");
        if (item) env->targets.target_elasticity = (float)item->value;
        item = dict_get_unsafe(kwargs, "target_tensile");
        if (item) env->targets.target_tensile = (float)item->value;
        item = dict_get_unsafe(kwargs, "target_toughness");
        if (item) env->targets.target_toughness = (float)item->value;
        item = dict_get_unsafe(kwargs, "target_lightweight");
        if (item) env->targets.target_lightweight = (float)item->value;
        item = dict_get_unsafe(kwargs, "max_solvation_kcal");
        if (item) env->targets.max_solvation_kcal = (float)item->value;
        item = dict_get_unsafe(kwargs, "min_wall_pressure_bar");
        if (item) env->targets.min_wall_pressure_bar = (float)item->value;
        item = dict_get_unsafe(kwargs, "max_molecular_weight");
        if (item) env->targets.max_molecular_weight = (float)item->value;
        item = dict_get_unsafe(kwargs, "min_valency");
        if (item) env->targets.min_valency = (int)item->value;
    }
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "r_thermo", log->r_thermo);
    dict_set(out, "r_elasticity", log->r_elasticity);
    dict_set(out, "r_tensile", log->r_tensile);
    dict_set(out, "r_toughness", log->r_toughness);
    dict_set(out, "r_lightweight", log->r_lightweight);
    dict_set(out, "p_wall", log->p_wall);
    dict_set(out, "omega_solv", log->omega_solv);
    dict_set(out, "valid_molecules", log->valid_molecules);
}
