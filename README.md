
## RMNA: A Neighbor Aggregation-Based Knowledge Graph Representation Learning Model Using Rule Mining

##### Our code is based on [Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs](https://github.com/deepakn97/relationPrediction)
This README is also based on it.

This repository contains a Pytorch implementation of RMNA. 
We use [AMIE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/) to obtains horn rules.
RMNA is a hierarchical neighbor aggregation model, which transforms valuable multi-hop neighbors into one-hop neighbors 
that are semantically similar to the corresponding multi-hop neighbors, so that the completeness of multi-hop neighbors can be ensured.


### Requirements
- [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)
- Java

Please download miniconda from above link and create an environment using the following command:

        conda env create -f pytorch35.yml

Activate the environment before executing the program as follows:

        source activate pytorch35
### Dataset
We used two datasets for evaluating our model. All the datasets and their folder names are given below.
- Freebase: FB15k-237
- Wordnet: WN18RR

### Rule Mining and Filtering

In the AMINE+ folder, we can generate mining rules by using the following command:

        java -jar amie_plus.jar [TSV file]

Without additional arguments AMIE+ thresholds using PCA confidence 0.1 and head coverage 0.01. You can change these default settings. See [AMIE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/amie/).
The available files generated and processed by AMIE are placed in the folder of the corresponding dataset named **new_triple**.  
### Training

**Parameters:**

`--data`: Specify the folder name of the dataset.

`--epochs_gat`: Number of epochs for gat training.

`--epochs_conv`: Number of epochs for convolution training.

`--lr`: Initial learning rate.

`--weight_decay_gat`: L2 reglarization for gat.

`--weight_decay_conv`: L2 reglarization for conv.

`--get_2hop`: Get a pickle object of 2 hop neighbors.

`--use_2hop`: Use 2 hop neighbors for training.  

`--partial_2hop`: Use only 1 2-hop neighbor per node for training.

`--output_folder`: Path of output folder for saving models.

`--batch_size_gat`: Batch size for gat model.

`--valid_invalid_ratio_gat`: Ratio of valid to invalid triples for GAT training.

`--drop_gat`: Dropout probability for attention layer.

`--alpha`: LeakyRelu alphas for attention layer.

`--nhead_GAT`: Number of heads for multihead attention.

`--margin`: Margin used in hinge loss.

`--batch_size_conv`: Batch size for convolution model.

`--alpha_conv`: LeakyRelu alphas for conv layer.

`--valid_invalid_ratio_conv`: Ratio of valid to invalid triples for conv training.

`--out_channels`: Number of output channels in conv layer.

`--drop_conv`: Dropout probability for conv layer.


### How to run

When running for first time, run preparation script with:

        $ sh prepare.sh

* **Freebase**

        $ python3 main.py --data ./data/FB15k-237/ --epochs_gat 2000 --epochs_conv 150  --get_2hop True --partial_2hop True --batch_size_gat 272115 --margin 1 --out_channels 50 --drop_conv 0.3 --output_folder ./checkpoints/fb/out/

* **Wordnet**

        $ python3 main.py --data ./data/WN18RR/--epochs_gat 3600 --epochs_conv 150 --get_2hop True --partial_2hop True