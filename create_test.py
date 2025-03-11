import os
import time
import argparse
import pandas as pd
from Bio import SeqIO
from Bio.SeqIO import FastaIO
import tensorflow as tf
from tensorflow.keras.models import load_model

from scripts.tetree import get_parent_nodes
from scripts.generate_data import get_kmer_data, get_oh_data
from scripts.classify_pipeline import get_labels, hi_classify


def get_parsed_args():
    parser = argparse.ArgumentParser(description="CREATE classify TEs using an attention-based hybrid CNN-RNN model.")

    parser.add_argument("-i", "--input_file", dest="input_file", 
                        required = True, help="Path to the input test sequences file.")
    parser.add_argument("-d", "--model_dir", dest="model_dir",
                        required = True, help="Directory containing trained models for classification.")
    parser.add_argument("-p", "--prob_thr", dest="prob_thr", type=float, default=0.90, 
                        help="Probability threshold for classifying a TE into a specific model.")
    parser.add_argument("-o", "--output_dir", dest="output_dir", default="./",
                        required = False, help="Directory to save the output files (default: current directory).")
    parser.add_argument("-k", "--k_mer", dest="k_mer", type=int, default=7,
                        required = False, help="Size of k-mers for feature extraction in CNN model.")
    parser.add_argument("-l", "--seq_len", dest="seq_len", type=int, default=600,
                        required = False, help="Length of the sequences extracted from both ends for RNN model.")

    args = parser.parse_args()
    return args


def main():
    args = get_parsed_args()

    input_file = args.input_file
    model_dir = args.model_dir
    prob_thr = float(args.prob_thr)
    output_dir = args.output_dir
    k = int(args.k_mer)
    l = int(args.seq_len)


    temp_res_dir = output_dir + "/temp/"
    if not os.path.exists(temp_res_dir):
        os.makedirs(temp_res_dir)

    # Transfer fasta data into input data
    print("\nStep 1: Transfer fasta data into input data ...")

    print("\t1) Transfer fasta data into k-mer frequency ...")
    X_kmer = get_kmer_data(input_file, k)
    X_kmer = X_kmer.reshape(X_kmer.shape[0], 1, pow(4, k), 1)
    X_kmer = X_kmer.astype("float64")

    print("\t2) Transfer fasta data into one-hot encoding ...")
    X_oh = get_oh_data(input_file, l)

    print("\nStep 2: Predict the probability for test sequences, the details are stored in the output_dir/temp/ dictionary ...")
    model_name_list =  [node for node in get_parent_nodes() if node not in ["Sub1", "Sub2"]]

    for model_name in model_name_list:
        print(f"Predicting for {model_name} model ...")
        model_file = f"{model_dir}/{model_name}"

        model = load_model(model_file)
        model.build(input_shape=([(X_oh.shape[0], 1, pow(4, k), 1), ()]))
        # model.summary()

        res = model.predict([X_kmer, X_oh], verbose=1)
        res = pd.DataFrame(res)
        res.columns = get_labels(model_name)
        res.to_csv(f"{temp_res_dir}/{model_name}_pred_prob.txt", sep="\t", header=True)

    print("\nStep 3: Calculate the hierarchical classification results and update the fasta file ...")
    true_labels = []
    for record in SeqIO.parse(input_file, "fasta"):
        true_label = record.id.split("#")[0]
        true_labels.append(true_label)

    hi_res, pred_labels = hi_classify(true_labels, prob_thr, temp_res_dir)
    print(f"hP={hi_res.hierarchical_precision} hR={hi_res.hierarchical_recall} hF={hi_res.hierarchical_f1}")
    
    records = []
    idx = 0
    for record in SeqIO.parse(input_file, "fasta"):
        record.description = f"{pred_labels[idx]}\t{record.description}"
        record.id = record.description
        records.append(record)
        idx += 1
    
    # Use FastIO method to write the fasta sequence in a single line
    new_data_file = f"{output_dir}/CREATE_pred.fasta"
    out_handle = FastaIO.FastaWriter(new_data_file, wrap=None)
    out_handle.write_file(records)
    return 0


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)

    start_time = time.time()

    main()

    end_time = time.time()
    run_time = end_time - start_time
    print(f"\nExecution time: {round(run_time, 3)} seconds.")
