# RSMBind Data

This directory contains the input data required to run RSMBind for RNA small-molecule binding-site prediction.

Due to the size of the datasets and precomputed embeddings, the complete data files are hosted externally and are not included directly in this GitHub repository.

## 1. Data Download

Download the required files and directories from the links below:

| Data | Description | Download |
|---|---|---|
| JSON files | RNA structure and nucleotide-level input features required for model evaluation | [Download JSON files](https://drive.google.com/drive/folders/1kIXIvOnFr3Mtkdk6m5BbbTeKz02g3W27?usp=sharing) |
| PDB files | RNA–small-molecule complex structures | [Download PDB files](ADD_LINK_HERE) |
| RNABERT embeddings | Precomputed 120-dimensional RNA language-model embeddings | [Download RNABERT embeddings](https://drive.google.com/drive/folders/1kIXIvOnFr3Mtkdk6m5BbbTeKz02g3W27?usp=sharing) |
| ERNIE-RNA embeddings | Precomputed 768-dimensional RNA language-model embeddings | [Download ERNIE-RNA embeddings](https://drive.google.com/drive/folders/1kIXIvOnFr3Mtkdk6m5BbbTeKz02g3W27?usp=sharing) |
| RNA-FM embeddings | Precomputed 640-dimensional RNA language-model embeddings | [Download RNA-FM embeddings](https://drive.google.com/drive/folders/1kIXIvOnFr3Mtkdk6m5BbbTeKz02g3W27?usp=sharing) |

## 2. Directory Organization

After downloading, organize the files as follows:

```text
data/
├── json/
│   └── RNA structure JSON files
│
├── pdb/
│   └── RNA–small-molecule complex structures
│
├── rnabert_embeddings/
│   └── Precomputed RNABERT embeddings
│
├── ernie_embeddings/
│   └── Precomputed ERNIE-RNA embeddings
│
└── rnafm_embeddings/
    └── Precomputed RNA-FM embeddings
```

## 3. Important Notes

- The embedding files must correspond to the RNA structures used for evaluation.
- Keep the original filenames when extracting and organizing the downloaded files, as the evaluation scripts use these filenames to locate the corresponding embeddings.
- Ensure that all required data directories are correctly specified when running the evaluation script.
- The embedding dimensions are 120 for RNABERT, 768 for ERNIE-RNA, and 640 for RNA-FM.

For instructions on installing dependencies and running RSMBind, refer to the main [README](../README.md).
