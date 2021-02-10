# coding: utf-8
# 2021/2/7 @ tongshiwei
from longling import path_append
from collections import defaultdict
from longling import json_load
from EduSim.utils.io_lib import load_ks_from_csv


def load_transition_matrix(filepath):
    meta_data = json_load(filepath)

    for i in range(len(meta_data)):
        if isinstance(meta_data[i], dict):
            converted_dict = {}
            for k, v_dict in meta_data[i].items():
                converted_dict[int(k)] = defaultdict(float)
                for key, value in v_dict.items():
                    converted_dict[int(k)][int(key)] = value
            meta_data[i] = converted_dict

    return meta_data


def load_initial_states(filepath):
    return json_load(filepath)


def load_configuration(filepath):
    return json_load(filepath)


def load_state_to_vector(filepath):
    return json_load(filepath)


def load_knowledge_structure(filepath):
    return load_ks_from_csv(filepath)


def load_environment_parameters(directory):
    return {
        "transition_matrix": load_transition_matrix(path_append(directory, "transition_matrix.json")),
        "configuration": load_configuration(path_append(directory, "configuration.json")),
        "knowledge_structure": load_knowledge_structure(path_append(directory, "knowledge_structure.csv")),
        "state2vector": load_state_to_vector(path_append(directory, "state2vector.json")),
        "initial_states": load_initial_states(path_append(directory, "initial_states.json")),
    }
