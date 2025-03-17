import pandas as pd

from .tetree import *
from .eva_metrics import HierarchicalMetric


# Get child labels by the model name
def get_labels(model_name):
    label_list = None
    if model_name == "TE":
        label_list = ["ClassI", "ClassII"]
    elif model_name == "ClassI":
        label_list = ["LTR", "Non-LTR"]
    elif model_name == "ClassII":
        label_list = ["Sub1", "Sub2"]
    elif model_name == "ERV":
        label_list = ["ERV1", "ERV2", "ERV3", "ERV4"]
    elif model_name == "LTR":
        label_list = ["Bel-Pao", "Copia", "ERV", "Gypsy"]
    elif model_name == "Non-LTR":
        label_list = ["DIRS", "LINE", "PLE", "SINE"]
    elif model_name == "LINE":
        label_list = ["CR1", "I", "Jockey", "L1", "R2", "RTE", "Rex1"]
    elif model_name == "SINE":
        label_list = ["ID", "SINE1/7SL", "SINE2/tRNA", "SINE3/5S"]
    elif model_name == "TIR":
        label_list = ["CACTA", "MULE", "PIF", "TcMar", "hAT"]
    return label_list

# Save the results of different models in a dictionary
def get_res(model_name_list, temp_res_dir):
    res_dict = {}
    model_name_list = [node for node in get_parent_nodes() if node not in ["Sub1", "Sub2"]]

    for model_name in model_name_list:
        # Calculate the mean probability for each model
        file = f"{temp_res_dir}/{model_name}_pred_prob.txt"
        pred_res_df = pd.read_csv(file, sep="\t", index_col=0)
        res_dict[model_name] = pred_res_df
    return res_dict


# Annotate TEs by probability threshold probability of LCPN
def thred_lcpn(idx, res_dict, prob_thr, label, leaf_node_list):
    if label == "Sub1":
        label = "TIR"
    max_prob_label = res_dict[label][idx:idx + 1].idxmax(axis=1).at[idx]

    if res_dict[label][idx:idx + 1][max_prob_label].at[idx] > prob_thr:
        if max_prob_label == "Sub2":
            max_prob_label = "Helitron"
        if max_prob_label not in leaf_node_list:
            yield from thred_lcpn(idx, res_dict, prob_thr, max_prob_label, leaf_node_list)
        else:
            yield max_prob_label
    else:
        yield label


def get_thr_lcpn(res_dict, prob_thr):
    pred_labels = []
    leaf_node_list = get_leaf_nodes()
    for idx in range(list(res_dict.values())[0].shape[0]):
        pred_label = next(thred_lcpn(idx, res_dict, prob_thr, "TE", leaf_node_list))
        pred_labels.append(pred_label)
    return pred_labels


def hi_classify(true_labels, prob_thr, temp_res_dir):
    # Get the true labels of the sequence
    true_hi_labels = []
    for true_label in true_labels:
        true_hi_label = find_node_path(true_label)
        true_hi_labels.append(true_hi_label.split("@"))

    # Get the predicted labels of the sequence
    model_name_list = [node for node in get_parent_nodes() if node not in ["Sub1", "Sub2"]]

    res_dict = get_res(model_name_list, temp_res_dir)
    pred_labels = get_thr_lcpn(res_dict, prob_thr)
    pred_hi_labels = [find_node_path(pred_label).split("@") for pred_label in pred_labels]
    hi_res = HierarchicalMetric(true_hi_labels, pred_hi_labels)
    return hi_res, pred_labels

