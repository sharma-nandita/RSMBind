# RSMBind: RNA Small-Molecule Binding-Site Prediction by Integrating RNA Language Models and Structural Geometry

RSMBind is a graph attention-based deep learning framework for predicting small-molecule binding sites at the nucleotide level in RNA structures. It integrates handcrafted RNA features, three-dimensional structural information, RNA secondary structure, and pretrained RNA language model embeddings to identify potential small-molecule binding nucleotides.

RSMBind employs an edge-aware Graph Attention Network (GATv2) to learn nucleotide-level representations from RNA structural graphs. The model incorporates embeddings from RNA-FM, RNABERT, and ERNIE-RNA alongside handcrafted features to improve binding-site prediction.

This repository provides the complete implementation of RSMBind, including data preprocessing scripts, precomputed RNA language model embeddings, processed datasets, trained five-fold model checkpoints, and model evaluation scripts.

## Repository Contents

```text
RSMBind/
│
├── data/
│   ├── csv_files/
│   │
│   ├── train_json_files/
│   ├── rb9_json/
│   ├── TE18_json/
│   ├── TL12_json/
│   ├── JL10_json/
│   │
│   ├── JL10_pdbs/
│   ├── te18_pdbs/
│   ├── TL12_pdbs/
│   │
│   ├── rnafm_embeddings_Hariboss_TR60_train/
│   ├── rnabert_embeddings_Hariboss_TR60_train/
│   ├── ernie_embeddings_Hariboss_TR60_train/
│   │
│   ├── rnafm_embeddings_rb9/
│   ├── rnabert_embeddings_rb9/
│   ├── ernie_embeddings_rb9/
│   │
│   ├── rnafm_embeddings_TE18/
│   ├── rnabert_embeddings_TE18/
│   ├── ernie_embeddings_TE18/
│   │
│   ├── rnafm_embeddings_TL12/
│   ├── rnabert_embeddings_TL12/
│   ├── ernie_embeddings_TL12/
│   │
│   ├── rnafm_embeddings_JL10/
│   ├── rnabert_embeddings_JL10/
│   └── ernie_embeddings_JL10/
│
├── models/
│   ├── hc_rnafm_no_sasa_fold1_best.pt
│   ├── hc_rnafm_no_sasa_fold2_best.pt
│   ├── hc_rnafm_no_sasa_fold3_best.pt
│   ├── hc_rnafm_no_sasa_fold4_best.pt
│   └── hc_rnafm_no_sasa_fold5_best.pt
│
├── train.py
├── dataset.py
├── GATv2Conv.py
├── evaluation.py
├── run_all_ablation.py
│
├── convert_all_pdbs_updated.py
├── pdb_to_final_json_updated.py
│
├── precompute_Ernie.py
├── precompute_rnabert.py
├── ERNIE_embedder.py
├── rnabert_embedder.py
│
└── sasa_calculate.ipynb
```

### Main Scripts

| File | Description |
|---|---|
| `train.py` | Model training using the prepared RNA graph datasets. |
| `dataset.py` | Dataset loading, feature processing, and graph construction. |
| `GATv2Conv.py` | Edge-aware GATv2 graph convolution implementation. |
| `evaluation.py` | Evaluation using the trained five-fold models. |
| `run_all_ablation.py` | Execution of different feature combinations for ablation experiments. |
| `convert_all_pdbs_updated.py` | Conversion of RNA PDB structures into JSON input files. |
| `pdb_to_final_json_updated.py` | Additional PDB-to-JSON preprocessing. |
| `precompute_Ernie.py` | ERNIE-RNA embedding generation. |
| `precompute_rnabert.py` | RNABERT embedding generation. |
| `ERNIE_embedder.py` | ERNIE-RNA embedding extraction. |
| `rnabert_embedder.py` | RNABERT embedding extraction. |
| `sasa_calculate.ipynb` | Calculation of solvent-accessible surface area (SASA) features. |

## Method Overview

RSMBind represents RNA structures as spatial graphs, where individual nucleotides are represented as nodes and their structural relationships are represented as edges.

The model integrates the following features:

- **Handcrafted RNA features:** Nucleotide-level structural and physicochemical descriptors.
- **RNA-FM:** Pretrained RNA sequence embeddings.
- **RNABERT:** Contextual RNA nucleotide embeddings.
- **ERNIE-RNA:** Pretrained RNA language model embeddings.
- **Spatial relationships:** Three-dimensional nucleotide relationships based on an 8 Å distance cutoff.
- **Secondary structure:** RNA base-pairing information.

The graph incorporates RNA backbone connectivity, RNA secondary-structure interactions, and spatial relationships. An edge-aware GATv2 architecture processes the resulting graph to predict the probability of small-molecule binding at each nucleotide.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RSMBind.git
cd RSMBind
```

Create and activate a Conda environment:

```bash
conda create -n rsmbind python=3.10
conda activate rsmbind
```

Install the required Python packages:

```bash
pip install torch
pip install torch-geometric
pip install numpy pandas scipy scikit-learn
pip install biopython biotite
pip install fair-esm
```

The appropriate PyTorch and PyTorch Geometric versions should be selected according to the CUDA environment and system configuration.

Additional dependencies may be required for the RNA language models and preprocessing scripts.

## Data Description

The `data/` directory contains the preprocessed datasets and corresponding RNA language model embeddings.

The repository includes:

- **Training dataset:** HARIBOSS_TR60 training data.
- **Independent test datasets:** RB9, TE18, TL12, and JL10.
- **JSON files:** Preprocessed nucleotide-level RNA structural and interaction information.
- **RNA language model embeddings:** Precomputed RNA-FM, RNABERT, and ERNIE-RNA embeddings stored as NumPy (`.npy`) files.
- **PDB structures:** RNA structural files included in the respective dataset directories.
- **CSV files:** Dataset metadata and associated nucleotide-level information.

The JSON files and pretrained embeddings are provided in the repository. Therefore, users can directly perform model evaluation without repeating the preprocessing and embedding-generation steps.

## Model Evaluation Using Pretrained Checkpoints

RSMBind provides five trained fold-specific checkpoints in the `models/` directory.

The final nucleotide-level prediction is obtained by averaging the predicted probabilities from the five fold models.

### 1. Activate the Environment

```bash
conda activate PT
```

Alternatively, activate the environment created during installation.

### 2. Evaluate on RB9

Run the following command from the main `RSMBind/` directory:

```powershell
python .\evaluation.py `
    --json_folder .\data\rb9_json\ `
    --rnabert_dir .\data\rnabert_embeddings_rb9\ `
    --ernie_dir .\data\ernie_embeddings_rb9\ `
    --rnafm_dir .\data\rnafm_embeddings_rb9\ `
    --model_dir .\models_\ `
    --out_csv .\output\rb9_output.csv
```

### 3. Evaluate on TE18

```powershell
python .\evaluation.py `
    --json_folder .\data\TE18_json\ `
    --rnabert_dir .\data\rnabert_embeddings_TE18\ `
    --ernie_dir .\data\ernie_embeddings_TE18\ `
    --rnafm_dir .\data\rnafm_embeddings_TE18\ `
    --model_dir .\models\ `
    --out_csv .\output\TE18_output.csv
```

### 4. Evaluate on TL12

```powershell
python .\evaluation.py `
    --json_folder .\data\TL12_json\ `
    --rnabert_dir .\data\rnabert_embeddings_TL12\ `
    --ernie_dir .\data\ernie_embeddings_TL12\ `
    --rnafm_dir .\data\rnafm_embeddings_TL12\ `
    --model_dir .\models\ `
    --out_csv .\output\TL12_output.csv
```

### 5. Evaluate on JL10

```powershell
python .\evaluation.py `
    --json_folder .\data\JL10_json\ `
    --rnabert_dir .\data\rnabert_embeddings_JL10\ `
    --ernie_dir .\data\ernie_embeddings_JL10\ `
    --rnafm_dir .\data\rnafm_embeddings_JL10\ `
    --model_dir .\models\ `
    --out_csv .\output\JL10_output.csv
```

**Note:** Create the output directory before running the evaluation commands if it does not already exist.

### Prediction Output

The evaluation script generates CSV files containing nucleotide-level prediction results.

Each nucleotide is assigned a predicted binding probability between 0 and 1. These probabilities can be used to rank nucleotides according to their predicted likelihood of participating in small-molecule binding.

## Model Training

RSMBind can also be trained using the provided HARIBOSS_TR60 training data and precomputed RNA language model embeddings.

### Training Command

From the main repository directory, execute:

```powershell
python .\train.py `
    --json_folder .\data\train_json_files\ `
    --save_path .\models_ `
    --ablation_name hc_rnafm_no_sasa `
    --rnafm_dir .\data\rnafm_embeddings_Hariboss_TR60_train\ `
    --no_sasa
```

The arguments are:

- `--json_folder`: Directory containing the preprocessed training JSON files.
- `--save_path`: Output location for the trained model checkpoints.
- `--ablation_name`: Specifies the feature configuration.
- `--rnafm_dir`: Directory containing the RNA-FM embeddings.
- `--no_sasa`: Disables SASA features.

The selected configuration, `hc_rnafm_no_sasa`, uses handcrafted RNA features and RNA-FM embeddings without SASA features.

The model is developed using five-fold cross-validation. The best checkpoint for each fold is selected based on validation MCC.

**Note:** The training command above reproduces the selected feature configuration. Other feature combinations can be investigated using the ablation pipeline.

## Ablation Experiments

The contribution of different input features can be investigated using the provided ablation script.

Run:

```bash
python run_all_ablation.py
```

The script generates the feature combinations used in the ablation experiments and executes the corresponding model training configurations.

## Data Preprocessing and Embedding Generation

For users who want to reproduce the preprocessing pipeline from RNA PDB structures, the repository also provides the structural preprocessing and embedding-generation scripts.

### 1. Calculate SASA Features

Open and execute:

```text
sasa_calculate.ipynb
```

### 2. Generate JSON Files

Use `convert_all_pdbs_updated.py` to convert RNA PDB structures and associated CSV metadata into JSON files.

Example for a dataset:

```bash
python convert_all_pdbs_updated.py \
    --pdb_dir PATH_TO_PDB_FILES \
    --csv PATH_TO_METADATA_CSV \
    --out_dir PATH_TO_OUTPUT_JSON \
    --max_len 438
```

The same preprocessing procedure is applied to the training and independent test datasets.

### 3. Generate RNA Language Model Embeddings

The following scripts are provided:

```bash
python precompute_Ernie.py
python precompute_rnabert.py
```

The generated embeddings are stored as `.npy` files and used during model training and evaluation.

RNA-FM embeddings are also required for the selected RSMBind configuration. Precomputed RNA-FM embeddings are included in the `data/` directory.

## Trained Model Checkpoints

The five trained RSMBind checkpoints are provided in the `models/` directory:

| Checkpoint | Description |
|---|---|
| `hc_rnafm_no_sasa_fold1_best.pt` | Fold 1 trained model |
| `hc_rnafm_no_sasa_fold2_best.pt` | Fold 2 trained model |
| `hc_rnafm_no_sasa_fold3_best.pt` | Fold 3 trained model |
| `hc_rnafm_no_sasa_fold4_best.pt` | Fold 4 trained model |
| `hc_rnafm_no_sasa_fold5_best.pt` | Fold 5 trained model |

These checkpoints correspond to the selected RSMBind configuration, `hc_rnafm_no_sasa`.

Users can directly use these checkpoints for independent dataset evaluation without retraining the model.

## Citation

If you use RSMBind in your research, please cite our work:

**[Nandita Sharma, Pralay Mitra, "RNA Small-Molecule Binding-Site Prediction by Integrating RNA Language Models and Structural Geometry"]**

## Contact
Nandita Sharma

email: nandita.sharma9494@kgpian.iitkgp.ac.in

For questions regarding RSMBind, implementation details, or reproducibility, please contact the authors.

## License

The license and usage conditions for the source code, datasets, and pretrained model checkpoints will be specified in the repository.
