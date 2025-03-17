import numpy as np
from Bio import SeqIO
from Bio.SeqIO import FastaIO
from sklearn.preprocessing import OneHotEncoder


# Convert undetermined nucleotides to A/T/C/G base
def convert_undetermined_base(seq):
    seq = seq.upper()
    seq = seq.replace("Y", "C")
    seq = seq.replace("D", "G")
    seq = seq.replace("S", "C")
    seq = seq.replace("R", "G")
    seq = seq.replace("V", "A")
    seq = seq.replace("K", "G")
    seq = seq.replace("N", "T")
    seq = seq.replace("H", "A")
    seq = seq.replace("W", "A")
    seq = seq.replace("M", "C")
    seq = seq.replace("X", "G")
    seq = seq.replace("B", "C")
    # new_seq = Seq(seq)
    return seq


# Sequence one-hot encoding
def seq2oh(seq, len_thre):
    if len(seq) >= len_thre:
        seq_1 = list(seq)[0:len_thre//2]
        seq_2 = list(seq)[-len_thre//2:]
        seq = seq_1 + seq_2
    else:
        seq_1 = list(seq)[0:len(seq)//2]
        seq_2 = list(seq)[-len(seq)//2:]
        seq = seq_1 + [0] * (len_thre - len(seq)) + seq_2
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array(["A", "C", "G", "T"]).reshape(-1, 1))
    seq_encode = enc.transform(np.array(seq).reshape(-1, 1)).toarray()
    return seq_encode


def nucle2num(nucleotide):
    nucleotide = nucleotide.upper()
    if nucleotide == "A":
        return 0
    elif nucleotide == "G":
        return 1
    elif nucleotide == "C":
        return 2
    elif nucleotide == "T":
        return 3
    else:
        raise ValueError(f"Invalid nucleotide: {nucleotide}")


# Sequence kmer encoding
def seq2kmer(seq, k):
    b = 4 # base
    h = 0 # hash value
    hash_dict = {key: 0 for key in range(4 ** k)}
    if len(seq) >= k:
        # Initialize hash value
        for i in range(k):
            h = h * b + nucle2num(seq[i])
        hash_dict[h] = 1
        # Calculate frequency of remaining k-mers
        for i in range(k, len(seq)):
            h = (h - nucle2num(seq[i-k]) * b**(k-1)) * b + nucle2num(seq[i])
            hash_dict[h] += 1
    else:
        raise ValueError(f"Invalid k-mer size: {k}")
    return hash_dict


def get_kmer_data(data_file, k):
    kmer = []
    for record in SeqIO.parse(data_file, "fasta"):
        seq = convert_undetermined_base(str(record.seq))
        kmer.append(list(seq2kmer(seq, k).values()))
    X_kmer = np.array(kmer)
    return X_kmer


def get_oh_data(data_file, len_thre):
    one_hot = []
    for record in SeqIO.parse(data_file, "fasta"):
        seq = convert_undetermined_base(str(record.seq))
        one_hot.append(seq2oh(seq, len_thre))
    X_oh = np.array(one_hot)
    return X_oh


def get_label_data(data_file):
    labels = []
    for record in SeqIO.parse(data_file, "fasta"):
        labels.append(record.description.split("#")[-1])
    enc = OneHotEncoder()
    y = enc.fit_transform(np.array(labels).reshape(-1, 1)).toarray()
    col_name = enc.categories_
    return y, col_name[0]




def split_data_by_label(model_name_list, data_file, model_data_dir):
    for model_name in model_name_list:
        record_list = []
        for record in SeqIO.parse(data_file, "fasta"):
            if model_name == "TE":
                record.description = record.description.split("|")[1].split("@")[1]
                record.id = record.description
                record_list.append(record)
            elif model_name == "ClassI":
                if "@ClassI@" in record.description:
                    if "@LTR" in record.description:
                        record.description = "LTR"
                    else:
                        record.description = "Non-LTR"
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "LTR":
                if "@LTR@" in record.description:
                    record.description = record.description.split("|")[1].split("@")[3]
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "Non-LTR":
                if "@Non-LTR@" in record.description:
                    record.description = record.description.split("|")[1].split("@")[3]
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "LINE":
                if "@LINE@" in record.description:
                    record.description = record.description.split("|")[1].split("@")[4]
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "SINE":
                if "@SINE@" in record.description:
                    record.description = record.description.split("|")[1].split("@")[4]
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "ERV":
                if "@ERV@" in record.description:
                    record.description = record.description.split("|")[1].split("@")[4]
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "ClassII":
                if "@ClassII@" in record.description:
                    if "@Sub1" in record.description:
                        record.description = "Sub1"
                    else:
                        record.description = "Sub2"
                    record.id = record.description
                    record_list.append(record)
            elif model_name == "TIR":
                if "@TIR@" in record.description:
                    record.description = record.description.split("|")[1].split("@")[4]
                    record.id = record.description
                    record_list.append(record)

        print(f"{model_name} contains {len(record_list)} sequences")
        # Use FastIO method to write the fasta sequence in a single line
        input_model_data_file = f"{model_data_dir}/{model_name}.fasta"
        if len(record_list) != 0:
            out_handle = FastaIO.FastaWriter(input_model_data_file, wrap=None)
            out_handle.write_file(record_list)
    return 0
