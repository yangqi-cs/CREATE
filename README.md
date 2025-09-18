# CREATE

## Introduction
CREATE leverages the advantages of <ins>**C**</ins>onvolutional neural networks and <ins>**R**</ins>ecurrent Neural N<ins>**E**</ins>tworks, combined with an <ins>**A**</ins>ttention mechanism, for efficient hierarchical classification of <ins>**T**</ins>ransposable <ins>**E**</ins>lements (TEs).

<div align=center>
<img src="https://github.com/user-attachments/assets/d2dd69f7-2dc0-4670-9af3-8c8c5f120de8" width="80%">
</div>

## Installation
Follow the steps below to set up CREATE:
#### 1. Clone the Repository

Retrieve the latest version of CREATE from the GitHub repository:
  
```
git clone https://github.com/yangqi-cs/CREATE.git
cd CREATE
```
#### 2. Set Up the Virtual Environment

Create and activate the virtual environment using conda:
```
conda env create -f env.yml
conda activate CREATE_env
```
#### 3. (Optional) Download the Pre-trained Model Directory
 
If you need the pre-trained model directory, download it from one of the following sources: <br>
- **Google drive:**  [model.tar.gz](https://drive.google.com/file/d/1OWzmD5sMM7kAJjKmq0LS3HibqzNVFThe/view?usp=sharing) <br>
- **Tencent cloud drive:** [model.tar.gz](https://share.weiyun.com/WNjtZPId)

After downloading, extract the model files into the ./model directory:
```
tar -zxvf model.tar.gz
```

## Usage
- #### create_train.py
Train models for TE classification. Available options:
```
-h, --help                  Display this help message and exit
-i, --input_file            Path to the input training sequences file
-o, --output_dir            Directory to save the output files
-m, --model_name            Specify the TE model for training
-k, --k_mer                 Size of k-mers for feature extraction in CNN model
-l, --seq_len               Length of the sequences extracted from both ends for the RNN model
-sr, --save_res             Whether to save predicted probabilities and classification report
-sm, --save_model           Whether to save the trained model
```
- #### create_test.py
Test trained models for TE classification. Available options:
```
-h, --help                  Display this help message and exit
-i, --input_file            Path to the input test sequences file
-d, --model_dir             Directory containing trained models for classification
-p, --prob_thr              Probability threshold for classifying a TE into a specific model
-o, --output_dir            Directory to save the output files (default: current directory)
-k, --k_mer                 Size of k-mers for feature extraction in CNN model
-l, --seq_len               Length of the sequences extracted from both ends for the RNN model
```

## Examples
By default, CREATE is designed to run on <ins>**GPU**</ins> rather than CPU. If you want to specify a particular GPU, please modify the ```create_train.py``` or ```create_test.py``` scripts by updating ```os.environ["CUDA_VISIBLE_DEVICES"] = "specific_gpu_index"``` before running the program. Replace ```"specific_gpu_index"``` with the target GPU ID.

- #### Training the Model

To train the model for a specific TE group dataset, ensure that the FASTA sequence names represent the subclass labels for that TE group (e.g., for the SINE model, they should be ID, SINE1/7SL, SINE2/tRNA, and SINE3/5S). Use the following command:
```
python create_train.py -i ./demo_data/SINE.fasta -m SINE
```

Alternatively, if your FASTA data is comprehensive and covers multiple TE groups, verify that the sequence names in the input FASTA file follow the format: ```ID|Class@Sub_class@Order@Superfamily|Species_type```. Then, you can train models according to the hierarchical classification structure of TEs using the command below:
```
python create_train.py -i ./demo_data/all_te.fasta -m All -sr -sm
```
<sub> **NOTE:** The primary difference between these two commands is the -m option. If the -m option is set to "All" (as in the second command) or omitted, the FASTA file will be divided into separate files according to family labels. Subsequently, classifiers will be trained for each parent node in the TE hierarchical structure. </sub>


- #### Testing the Model

The testing phase automatically performs the hierarchical classification process. Ensure that your test set contains only TE sequences and that the sequence names follow the format: ```Specific_TE_type#Description``` (e.g., ```Gypsy#Os0016_LTR|LTR/Gypsy```). If you are unsure whether your test set is a high-quality TE collection,  we strongly recommend first training a classifier to distinguish TEs from non-TEs based on your data type. You can then apply the specific TE model to predict the types of the identified TE sequences.

To evaluate the performance of the trained models on a new dataset, use the following command:
```
python create_test.py -i ./demo_data/test_data.fasta -d ./model/
```

## FAQs
Q: When executing the command ```conda env create -f env.yml```, you may encounter the error "CondaEnvException: Pip failed."

> **Solution 1 (Modify the index URL):** (1) Open the ```env.yml``` file and uncomment (remove the # symbol) the line that starts with --index-url. (2) Recreate the virtual environment using the following command: ```conda env create -f env.yml```.
> 
> **Solution 2 (Address pip and TensorFlow issues):** (1) Open the ```env.yml``` file and comment out the following lines (add a # symbol at the beginning of the lines): ```- pip:``` and ```- tensorflow-gpu==2.4.0```. (2) Create the virtual environment: ```conda env create -f env.yml```. (3) Activate the virtual environment: ```conda activate CREATE_env```. (4) Manually install the tensorflow-gpu package: ```pip install tensorflow-gpu==2.4.0```.


## Contact
:e-mail: **Yang Qi** (yang.qi@mail.nwpu.edu.cn)

School of Computer Science, Northwestern Polytechnical University, Xi’an, Shaanxi 710072, China
