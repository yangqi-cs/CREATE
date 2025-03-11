import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from scripts.tetree import get_parent_nodes
from scripts.generate_data import get_kmer_data, get_oh_data, get_label_data, split_data_by_label
from scripts.attention_model import train_attn_model


def get_parsed_args():
    parser = argparse.ArgumentParser(description="CREATE classify TEs using an attention-based hybrid CNN-RNN model.")

    parser.add_argument("-i", "--input_file", dest="input_file",
                        required = True, help="Path to the input training sequences file.")
    parser.add_argument("-o", "--output_dir", dest="output_dir", default="./", 
                        required = False, help="Directory to save the output files.")
    parser.add_argument("-m", "--model_name", dest="model_name", default=None, 
                        choices = ["All", "SINE", "ERV", "LINE", "Non-LTR", "TIR", "LTR", "ClassII", "ClassI", "TE"],
                        required = False, help="Specify the TE model for training.")
    parser.add_argument("-k", "--k_mer", dest="k_mer", type=int, default=7, 
                        required = False, help="Size of k-mers for feature extraction in CNN model.")
    parser.add_argument("-l", "--seq_len", dest="seq_len", type=int, default=600,
                        required = False, help="Length of the sequences extracted from both ends for RNN model.")
    parser.add_argument("-sr", "--save_res", dest="save_res", default=False, const=True, action="store_const",
                        required = False, help="Whether to save predicted probabilities and classification report.")
    parser.add_argument("-sm", "--save_model", dest="save_model", default=False, const=True, action="store_const",
                        required = False, help="Whether to save the trained model.")

    args = parser.parse_args()
    return args


def train_model(model_name, k, l, data_file, output_dir, save_res, save_model):
    print(f"Step 1: Transfer fasta data into input data for {model_name} ...")

    print("\t1) Transfer fasta data into k-mer frequency ...")
    X_kmer = get_kmer_data(data_file, k)
    X_kmer = X_kmer.reshape(X_kmer.shape[0], 1, pow(4, k), 1)
    X_kmer = X_kmer.astype("float64")

    print("\t2) Transfer fasta data into one-hot encoding ...")
    X_oh = get_oh_data(data_file, l)

    print("\t3) Transfer label data into one-hot encoding ...")
    y, col_name = get_label_data(data_file)

    print(f"Step 2: Train {model_name} model ...")
    X_oh_train, X_oh_test, X_kmer_train, X_kmer_test, y_train, y_test = train_test_split(X_oh, X_kmer, y, stratify=y, test_size=0.1)
    model = train_attn_model(X_oh_train, X_oh_test, X_kmer_train, X_kmer_test, y_train, y_test, col_name, k, l)

    print(f"Step 3: Test {model_name} model ...")
    # Evaluate the model on test data
    score = model.evaluate([X_kmer_test, X_oh_test], y_test, verbose=0)
    print("score = " + str(score))
    y_pred = np.argmax(model.predict([X_kmer_test, X_oh_test]), axis=-1)
    y_pred = keras.utils.to_categorical(y_pred, len(set(col_name)))
    pred_repo = classification_report(y_test, y_pred, target_names=col_name, digits=4)
    print(pred_repo)
    

    # Save result
    if save_res:
        res_output_dir = f"{output_dir}/res/"
        if not os.path.exists(res_output_dir):
            os.mkdir(res_output_dir)
        res = np.hstack((y_test, model.predict([X_kmer_test, X_oh_test])))
        res = pd.DataFrame(res, columns=list(np.concatenate((col_name, col_name))))
        res.to_csv(f"{res_output_dir}/{model_name}_prob.txt", sep="\t", header=True)

        repo = classification_report(y_test, y_pred, target_names=col_name, output_dict=True, digits=4)
        df = pd.DataFrame(repo).transpose()
        df.to_csv(f"{res_output_dir}/{model_name}_repo.txt", sep="\t", index=True)
    
    # Save model
    if save_model:
        model_output_dir = f"{output_dir}/model/"
        if not os.path.exists(model_output_dir):
            os.mkdir(model_output_dir)
        model.save(f"{model_output_dir}/{model_name}")    
    return 0



def main():
    args = get_parsed_args()

    if args.input_file is None:
        print("Please provide the input sequences file!")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    k_mer = int(args.k_mer)
    seq_len = int(args.seq_len)
    input_file = args.input_file
    output_dir = args.output_dir
    save_res = args.save_res
    save_model = args.save_model

    if args.model_name is not None and args.model_name != "All":
        model_name = args.model_name
        print("[1/1]")
        train_model(model_name, k_mer, seq_len, input_file, output_dir, save_res, save_model)
    elif args.model_name == "All":
            print("Step 0: Divide the fasta data into different dataset by TE model name ...")
            model_name_list =  [node for node in get_parent_nodes() if node not in ["Sub1", "Sub2"]]
            model_data_dir = f"{output_dir}/model_data"
            if not os.path.exists(model_data_dir):
                os.mkdir(model_data_dir)
            split_data_by_label(model_name_list, input_file, model_data_dir)
            for idx, model_name in enumerate(model_name_list, start=1):
                input_model_data_file = f"{model_data_dir}/{model_name}.fasta"

                if not os.path.exists(input_model_data_file):
                    print("WARNING: The ", model_name, " model does not have any sequences, so this file is not generated.")
                else:
                    print(f"[{idx}/{len(model_name_list)}]")
                    train_model(model_name, k_mer, seq_len, input_model_data_file, output_dir, save_res, save_model)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), model_name, " DONE!\n")
    else:
        print("You have not entered the TE model name through the parameter '-n' or '--model_name'.\n"
              "The model will be trained according to the hierarchical classification structure in CREATE.")
        in_con = input("Please confirm (y/n): ")
        if in_con == "y" or in_con == "Y":
            print("Step 0: Divide the fasta data into different dataset by TE model name ...")
            model_name_list =  [node for node in get_parent_nodes() if node not in ["Sub1", "Sub2"]]
            model_data_dir = f"{output_dir}/model_data"
            if not os.path.exists(model_data_dir):
                os.mkdir(model_data_dir)
            split_data_by_label(model_name_list, input_file, model_data_dir)
            for idx, model_name in enumerate(model_name_list, start=1):
                input_model_data_file = f"{model_data_dir}/{model_name}.fasta"

                if not os.path.exists(input_model_data_file):
                    print("WARNING: The ", model_name, " model does not have any sequences, so this file is not generated.")
                else:
                    print(f"[{idx}/{len(model_name_list)}]")
                    train_model(model_name, k_mer, seq_len, input_model_data_file, output_dir, save_res, save_model)
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())), model_name, " DONE!\n")

        elif in_con == "n" or in_con == "N":
            print("Please provide an appropriate TE model name through the parameter '-n' or '--model_name'. Thank you!")
            return
        else:
            print("Oops! We did not receive the proper instruction. Please ensure that the input must be 'y' or 'n'!")
            return
        

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)


    start_time = time.time()

    main()

    end_time = time.time()
    run_time = end_time - start_time
    print("Execution time:", round(run_time, 3), "seconds.")

