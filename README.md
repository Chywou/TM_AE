# Readme TM

Author: Laetitia Guidetti

## Project Overview

This project aims to explore the use of autoencoders for anomaly detection and data compression in the context of astronomical observations with the SST-1M telescope. The project includes several autoencoder models, each with a different architecture, as well as experiments using Random Forests for classification and regression.

## Prerequisites

This project is designed to be run on the HPC server **baobab** of the University of Geneva (https://www.unige.ch/eresearch/en/services/hpc/).

The main prerequisites are:
- access to the **baobab** server
- access to the shared directory containing the data (group GL_S_PHYS_HPC_SST1M)

## Data Description

Data used in this project are not included in this repository due to their large size. They are stored on the **baobab** server and can be accessed via a shared directory.

The structure of the data directory is as follows:

```text
data/
├── gammas_diffuse/
│   ├── reduce_train/
│   └── reduce_test/
├── gammas_point/
│   ├── reduce_train/
│   └── reduce_test/
└── protons_diffuse/
    ├── reduce_train/
    └── reduce_test/
``` 

Each subfolder `reduce_train/` and `reduce_test/` contains 16 HDF5 files. Each HDF5 file contains data from a variable number of simulated events.

The autoencoders used for anomaly detection are trained on data from `protons_diffuse/reduce_train/` and evaluated on data from `gammas_diffuse/reduce_test/` and `protons_diffuse/reduce_test/`.

The classification RFs are trained on data from `protons_diffuse/reduce_train/` and `gammas_diffuse/reduce_train/`, and evaluated on data from `protons_diffuse/reduce_test/` and `gammas_diffuse/reduce_test/`.

The regression RFs are trained and evaluated on data from `gammas_diffuse/reduce_train/` and evaluated on `gammas_point/reduce_test/`.

## Repository structure

The repository is structured as follows:

- `paper.pdf`: TODO: the final report in PDF format. It contains a detailed description of the models, experiments, results, and conclusions.
- `ae_gpu.yml`: Conda configuration file used to create the containerized environment on the **baobab** server.
- `train_ae.sh`: bash script used to submit autoencoder training jobs on the **baobab** server via the SLURM job manager.
- `server_report/`: contains all the results and scripts executed on the **baobab** server used to write the report. This folder is structured as follows:
    - `data_analysis.ipynb`: Jupyter notebook containing preliminary data analysis and visualization.
    - `config.yaml`: configuration file used to define the parameters of the models and experiments, common to all models.
    - `utils.py`: file containing utility functions used in various scripts.
    - `ae_flat/`: contains the results and scripts for the Flat AE model.
    - `ae_square/`: contains the results and scripts for the Square AE model.
    - `ae_graph/`: contains the results and scripts for the Graph AE model.
    - `ae_hexa/`: contains the results and scripts for the Hexa AE model.
    These subfolders contain subfolders for each experiment conducted, their name directly refers to the models described in the report. Each subfolder contains the scripts used for training and evaluating the corresponding model, typically:
        - `ae_train.py`: script to train the AE model with the parameters defined in the `config.yaml` file. All necessary functions are directly present in this script.
        - `ae_show.ipynb`: notebook to visualize the training and testing results of the AE model.
        - `autoencoder_xxxxxxxx_xxxxxx.pth`: file containing the weights of the trained model, it is loaded in the `ae_show.ipynb` script to evaluate the model. The data used for training and evaluation are on the **baobab** server.
        - `errors_gammas_xxxxxxxx_xxxxxx.csv`: CSV file containing the reconstruction errors on the gamma test data for the trained AE model.
        - `errors_protons_xxxxxxxx_xxxxxx.csv`: CSV file containing the reconstruction errors on the proton test data for the trained AE model.
        - `energies_gammas_xxxxxxxx_xxxxxx.csv`: CSV file containing the energies of the gamma events
        - `energies_protons_xxxxxxxx_xxxxxx.csv`: CSV file containing the energies of the proton events
    - `ae_compress/`: contains the results and scripts for the experiments related to Random Forests. It contains the following files:
        - `config_2.yaml`: specific configuration file for the autoencoders used for data compression before training the Random Forests.
        - `config_3.yaml`: specific configuration file for the Random Forest experiments.
        - `rf_all_analysis.ipynb`: notebook to compare the performance of different random forests trained with and without compression by autoencoder.
        - `ae_hexa.../`: subfolders for each autoencoder experiment used for data compression before training the Random Forests, similar to the structure described earlier.
        - `rf_ae..../`: subfolders for each Random Forest experiment trained on data compressed by an autoencoder, containing the corresponding scripts and results. Each subfolder typically contains:
            - `rf_cl.py`: script to train the classification Random Forest.
            - `rf_rg.py`: script to train the regression Random Forest.
            - `rf.ipynb`: notebook to visualize the performance of the trained Random Forest.
            - `autoencoder_model.pth`: file containing the weights of the autoencoder used for data compression.
            - `rf_classifier.pkl`: file containing the trained classification Random Forest model (not included in the repository).
            - `rf_classifier_improved.pkl`: file containing the classification Random Forest model trained with features calculated from the autoencoder outputs (not included in the repository).
            - `rf_regressor.pkl`: file containing the trained regression Random Forest model (not included in the repository).
            - `rf_regressor_improved.pkl`: file containing the regression Random Forest model trained with features calculated from the autoencoder outputs (not included in the repository).
            - `rf_results_clf.csv`: CSV file containing the classification results of the Random Forest.
            - `rf_results_rg.csv`: CSV file containing the regression results of the Random Forest.

Important note: The trained Random Forest models (`.pkl` files) are not included in the repository due to their large size. They can be generated by running the corresponding training scripts. If you wish to obtain these files directly, please contact the author.

## Create container environment

On the **baobab** server, a containerized environment was created to ensure the reproducibility of experiments. The container is based on an Ubuntu 22.04 image and uses Conda to manage Python dependencies.

The steps to create the containerized environment are as follows:
```bash
ml purge
module load GCCcore/13.3.0 cotainr
cotainr build ctlearnenv.sif --base-image=docker://ubuntu:22.04 --accept-licenses --conda-env=ae_gpu.yml -v
```

This command creates a container named `ctlearnenv.sif` using the `ae_gpu.yml` file to install all necessary dependencies. This container is then used in the `train_ae.sh` script to execute model training tasks on the **baobab** server.

## Submitting training jobs

Training jobs for the autoencoders are submitted to the SLURM job manager on the **baobab** server using the `train_ae.sh` script. This script specifies the resources required for training, such as runtime, memory, and GPU type.

To submit a job, use the following command:
```bash
sbatch train_ae.sh
```

The following points should be modified in the `train_ae.sh` script if the directory structure or configurations change:
- The `CONTAINER` variable should point to the path of the previously created container.
- The `BIND` variable should be set to bind the directory containing the data and scripts to the container.
- The `apptainer exec` command should be adjusted to execute the appropriate training script with the correct configuration file.

In the YAML configuration files, the data paths should also be updated if necessary.

## Reproducing an experiment

To reproduce a specific experiment, follow these steps:
1. Create the containerized environment using the `ae_gpu.yml` file
2. Check the data paths in the YAML configuration files
3. Adapt the `train_ae.sh` script with the path to the desired training script and configuration file
4. Submit the job with `sbatch train_ae.sh`
5. Analyze the results and visualize the performance using the `ae_show.ipynb` notebook provided in the corresponding experiment subfolder.