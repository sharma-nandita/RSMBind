import os
import re
import argparse
from collections import defaultdict

import torch
import pandas as pd
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score
)

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from dataset import RNALigandJSONDataset
from GATv2Conv import RNAOnlyGNN


# ============================================================
# Fixed thresholds for EXTERNAL TEST evaluation
#
# IMPORTANT:
# No threshold optimization is performed on the test set.
# ============================================================

FIXED_THRESHOLDS = [0.4, 0.5, 0.6]

EXPECTED_FOLDS = [1, 2, 3, 4, 5]
# EXPECTED_FOLDS = [1, 2, 3]

# ============================================================
# Metrics at a fixed threshold
# ============================================================

def metrics_at_threshold(probs_all, labels_all, threshold):

    preds = (probs_all >= threshold).astype(int)

    return {
        "MCC": matthews_corrcoef(
            labels_all,
            preds
        ),

        "Precision": precision_score(
            labels_all,
            preds,
            zero_division=0
        ),

        "Recall": recall_score(
            labels_all,
            preds,
            zero_division=0
        ),

        "F1": f1_score(
            labels_all,
            preds,
            zero_division=0
        ),

        "Accuracy": accuracy_score(
            labels_all,
            preds
        )
    }


# ============================================================
# Evaluate ONE probability vector
#
# No threshold sweep.
# Only fixed thresholds.
# ============================================================

def evaluate_probabilities(probs_all, labels_all):

    results = {
        "ROC_AUC": roc_auc_score(
            labels_all,
            probs_all
        ),

        "PR_AUC": average_precision_score(
            labels_all,
            probs_all
        )
    }

    for threshold in FIXED_THRESHOLDS:

        threshold_metrics = metrics_at_threshold(
            probs_all,
            labels_all,
            threshold
        )

        threshold_string = str(threshold).replace(".", "_")

        for metric_name, value in threshold_metrics.items():

            results[
                f"t{threshold_string}_{metric_name}"
            ] = value

    return results


# ============================================================
# Classify checkpoint
# ============================================================

def classify_checkpoint(ckpt, model_file):

    # --------------------------------------------------------
    # Preferred: checkpoint metadata
    # --------------------------------------------------------

    if isinstance(ckpt, dict):

        if "fold" in ckpt:

            return "fold", int(ckpt["fold"])

        if "best_fold" in ckpt:

            return "best_overall", int(
                ckpt["best_fold"]
            )

    # --------------------------------------------------------
    # Filename fallback
    # --------------------------------------------------------

    filename = os.path.basename(
        model_file
    ).lower()

    match = re.search(
        r"fold[_\-]?(\d+)",
        filename
    )

    if match:

        return "fold", int(
            match.group(1)
        )

    if "best" in filename:

        return "best_overall", None

    return "unknown", None


# ============================================================
# Get ablation name
# ============================================================

def get_ablation_name(ckpt, model_file):

    if isinstance(ckpt, dict):

        name = ckpt.get(
            "ablation_name",
            None
        )

        if name is not None:

            return str(name)

    # Filename fallback
    filename = os.path.splitext(
        os.path.basename(model_file)
    )[0]

    # Remove fold information
    filename = re.sub(
        r"_?fold[_\-]?\d+",
        "",
        filename,
        flags=re.IGNORECASE
    )

    # Remove best suffix
    filename = re.sub(
        r"_?best$",
        "",
        filename,
        flags=re.IGNORECASE
    )

    return filename


# ============================================================
# Load model from checkpoint
# ============================================================

def load_model_from_checkpoint(
    ckpt,
    dataset,
    device
):

    # --------------------------------------------------------
    # New checkpoint format
    # --------------------------------------------------------

    if (
        isinstance(ckpt, dict)
        and "model_state" in ckpt
    ):

        state_dict = ckpt[
            "model_state"
        ]

        in_dim = ckpt.get(
            "in_dim",
            dataset[0].x.shape[1]
        )

        hidden_dim = ckpt.get(
            "hidden_dim",
            128
        )

        num_layers = ckpt.get(
            "num_layers",
            2
        )

        dropout = ckpt.get(
            "dropout",
            0.0
        )

    # --------------------------------------------------------
    # Old pure state_dict format
    # --------------------------------------------------------

    else:

        state_dict = ckpt

        in_dim = dataset[0].x.shape[1]

        hidden_dim = 128

        dropout = 0.0

        conv_keys = [
            key
            for key in state_dict
            if key.startswith("convs.")
        ]

        if len(conv_keys) == 0:

            raise RuntimeError(
                "Could not determine num_layers "
                "from checkpoint."
            )

        layer_ids = set(
            int(key.split(".")[1])
            for key in conv_keys
        )

        num_layers = max(
            layer_ids
        ) + 1

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = RNAOnlyGNN(
        in_node_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,

        use_rnabert=ckpt.get(
            "use_rnabert",
            False
        ) if isinstance(ckpt, dict) else False,

        use_ernie=ckpt.get(
            "use_ernie",
            False
        ) if isinstance(ckpt, dict) else False,

        use_rnafm=ckpt.get(
            "use_rnafm",
            False
        ) if isinstance(ckpt, dict) else False,

    ).to(device)

    model.load_state_dict(
        state_dict
    )

    model.eval()

    model_info = {
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout
    }

    return model, model_info


# ============================================================
# Generate predictions from ONE model
# ============================================================

@torch.no_grad()
def predict_probabilities(
    model,
    loader,
    device
):

    model.eval()

    probs_all = []
    labels_all = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(batch)

        probs = torch.sigmoid(
            logits
        )

        probs_all.append(
            probs.detach().cpu()
        )

        labels_all.append(
            batch.y.detach().cpu()
        )

    probs_all = torch.cat(
        probs_all
    ).numpy()

    labels_all = torch.cat(
        labels_all
    ).numpy()

    return probs_all, labels_all


# ============================================================
# Collect all checkpoints
# ============================================================

def collect_checkpoints(
    model_dir,
    device
):

    model_files = []

    for root, _, files in os.walk(
        model_dir
    ):

        for file in files:

            if file.endswith(".pt"):

                model_files.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    model_files = sorted(
        model_files
    )

    print(
        f"Found {len(model_files)} "
        f"checkpoint(s)"
    )

    # --------------------------------------------------------
    # Group by ablation
    # --------------------------------------------------------

    groups = defaultdict(list)

    skipped = []

    for model_file in model_files:

        ckpt = torch.load(
            model_file,
            map_location=device,
            weights_only=False
        )

        model_type, fold_num = (
            classify_checkpoint(
                ckpt,
                model_file
            )
        )

        # ----------------------------------------------------
        # Only actual fold checkpoints
        # ----------------------------------------------------

        if model_type != "fold":

            skipped.append(
                (
                    model_file,
                    model_type
                )
            )

            continue

        ablation_name = get_ablation_name(
            ckpt,
            model_file
        )

        groups[
            ablation_name
        ].append(
            (
                fold_num,
                model_file,
                ckpt
            )
        )

    # --------------------------------------------------------
    # Print grouping information
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("CHECKPOINT GROUPS")
    print("=" * 80)

    for ablation_name in sorted(
        groups.keys()
    ):

        fold_numbers = sorted(
            item[0]
            for item in groups[
                ablation_name
            ]
        )

        print(
            f"{ablation_name:35s} "
            f"{fold_numbers}"
        )

    print()
    print(
        f"Number of ablation groups: "
        f"{len(groups)}"
    )

    if len(skipped) > 0:

        print()
        print(
            f"Skipped {len(skipped)} "
            f"non-fold checkpoint(s)"
        )

    return groups


# ============================================================
# Build dataset for one ablation
#
# Embedding configuration is obtained from checkpoint metadata.
# ============================================================

def build_dataset(
    ckpt,
    args
):

    use_rnabert = ckpt.get(
        "use_rnabert",
        False
    )

    use_ernie = ckpt.get(
        "use_ernie",
        False
    )

    use_rnafm = ckpt.get(
        "use_rnafm",
        False
    )

    no_sasa = ckpt.get(
        "no_sasa",
        False
    )

    no_geometry = ckpt.get(
        "no_geometry",
        False
    )

    print(
        f"    RNABERT = {use_rnabert}, "
        f"ERNIE = {use_ernie}, "
        f"RNA-FM = {use_rnafm}"
    )

    print(
        f"    SASA = {not no_sasa}, "
        f"Geometry = {not no_geometry}"
    )

    dataset = RNALigandJSONDataset(

        args.json_folder,

        rnabert_dir=(
            args.rnabert_dir
            if use_rnabert
            else None
        ),

        ernie_dir=(
            args.ernie_dir
            if use_ernie
            else None
        ),

        rnafm_dir=(
            args.rnafm_dir
            if use_rnafm
            else None
        ),

        use_sasa=not no_sasa,

        use_geometry=not no_geometry
    )

    return dataset


# ============================================================
# Create DataLoader
# ============================================================

def build_loader(
    dataset,
    batch_size
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b:
            Batch.from_data_list(b)
    )


# ============================================================
# Evaluate ONE ABLATION GROUP
# ============================================================

def evaluate_ablation_group(
    ablation_name,
    fold_models,
    args,
    device
):

    print()
    print("#" * 90)
    print(
        f"ABLATION: {ablation_name}"
    )
    print("#" * 90)

    # --------------------------------------------------------
    # Sort folds
    # --------------------------------------------------------

    fold_models = sorted(
        fold_models,
        key=lambda x: x[0]
    )

    fold_numbers = [
        item[0]
        for item in fold_models
    ]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if fold_numbers != EXPECTED_FOLDS:

        raise RuntimeError(
            f"\nAblation '{ablation_name}' "
            f"does not contain exactly folds "
            f"1-5.\n"
            f"Found: {fold_numbers}\n"
            f"Expected: {EXPECTED_FOLDS}"
        )

    # --------------------------------------------------------
    # First checkpoint determines dataset configuration
    # --------------------------------------------------------

    first_ckpt = fold_models[0][2]

    dataset = build_dataset(
        first_ckpt,
        args
    )

    loader = build_loader(
        dataset,
        args.batch_size
    )

    print(
        f"    Number of RNAs/graphs: "
        f"{len(dataset)}"
    )

    print(
        f"    Total nucleotides: "
        f"{sum(data.num_nodes for data in dataset)}"
    )

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    fold_predictions = []

    fold_labels = []

    individual_results = []

    # ========================================================
    # Evaluate each fold
    # ========================================================

    for fold_num, model_file, ckpt in fold_models:

        print()
        print(
            "-" * 80
        )

        print(
            f"Fold {fold_num}"
        )

        print(
            os.path.basename(
                model_file
            )
        )

        # ----------------------------------------------------
        # Recreate model
        # ----------------------------------------------------

        model, model_info = (
            load_model_from_checkpoint(
                ckpt,
                dataset,
                device
            )
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        probs, labels = (
            predict_probabilities(
                model,
                loader,
                device
            )
        )

        # ----------------------------------------------------
        # Check labels
        # ----------------------------------------------------

        if len(fold_labels) > 0:

            if not np.array_equal(
                labels,
                fold_labels[0]
            ):

                raise RuntimeError(
                    f"Labels/order differ "
                    f"for fold {fold_num} "
                    f"in {ablation_name}."
                )

        fold_predictions.append(
            probs
        )

        fold_labels.append(
            labels
        )

        # ----------------------------------------------------
        # Individual fold metrics
        # ----------------------------------------------------

        metrics = evaluate_probabilities(
            probs,
            labels
        )

        row = {

            "ablation":
                ablation_name,

            "model":
                model_file,

            "type":
                "fold",

            "fold":
                fold_num,

            "use_rnabert":
                ckpt.get(
                    "use_rnabert",
                    False
                ),

            "use_ernie":
                ckpt.get(
                    "use_ernie",
                    False
                ),

            "use_rnafm":
                ckpt.get(
                    "use_rnafm",
                    False
                ),

            "no_sasa":
                ckpt.get(
                    "no_sasa",
                    False
                ),

            "no_geometry":
                ckpt.get(
                    "no_geometry",
                    False
                ),

            "num_layers":
                model_info[
                    "num_layers"
                ],

            "hidden_dim":
                model_info[
                    "hidden_dim"
                ],

            "dropout":
                model_info[
                    "dropout"
                ],

            "pos_weight":
                ckpt.get(
                    "pos_weight",
                    None
                ),

            "train_batch_size":
                ckpt.get(
                    "batch_size",
                    None
                )
        }

        row.update(
            metrics
        )

        individual_results.append(
            row
        )

        # ----------------------------------------------------
        # Print fold result
        # ----------------------------------------------------

        print(
            f"ROC-AUC = "
            f"{metrics['ROC_AUC']:.4f}"
        )

        print(
            f"PR-AUC  = "
            f"{metrics['PR_AUC']:.4f}"
        )

        for threshold in FIXED_THRESHOLDS:

            ts = str(
                threshold
            ).replace(
                ".",
                "_"
            )

            print(
                f"@ {threshold}: "
                f"MCC="
                f"{metrics[f't{ts}_MCC']:.4f}, "
                f"Precision="
                f"{metrics[f't{ts}_Precision']:.4f}, "
                f"Recall="
                f"{metrics[f't{ts}_Recall']:.4f}, "
                f"F1="
                f"{metrics[f't{ts}_F1']:.4f}, "
                f"Accuracy="
                f"{metrics[f't{ts}_Accuracy']:.4f}"
            )

    # ========================================================
    # Prediction matrix
    #
    # Shape:
    #
    # [5 folds, total nucleotides]
    # ========================================================

    prediction_matrix = np.stack(
        fold_predictions,
        axis=0
    )

    labels = fold_labels[0]

    # ========================================================
    # Fold average / STD
    # ========================================================

    fold_df = pd.DataFrame(
        individual_results
    )

    metric_columns = [
        "ROC_AUC",
        "PR_AUC"
    ]

    for threshold in FIXED_THRESHOLDS:

        ts = str(
            threshold
        ).replace(
            ".",
            "_"
        )

        for metric in [
            "MCC",
            "Precision",
            "Recall",
            "F1",
            "Accuracy"
        ]:

            metric_columns.append(
                f"t{ts}_{metric}"
            )

    avg_row = {

        "ablation":
            ablation_name,

        "model":
            "FOLD_AVERAGE",

        "type":
            "fold_average",

        "fold":
            "-"

    }

    std_row = {

        "ablation":
            ablation_name,

        "model":
            "FOLD_STD",

        "type":
            "fold_std",

        "fold":
            "-"

    }

    for column in metric_columns:

        avg_row[column] = (
            fold_df[column].mean()
        )

        std_row[column] = (
            fold_df[column].std()
        )

    # ========================================================
    # MEAN ENSEMBLE
    #
    # Average probability of five fold models
    # before thresholding.
    # ========================================================

    mean_probs = np.mean(
        prediction_matrix,
        axis=0
    )

    mean_metrics = (
        evaluate_probabilities(
            mean_probs,
            labels
        )
    )

    mean_row = {

        "ablation":
            ablation_name,

        "model":
            "ENSEMBLE_MEAN",

        "type":
            "ensemble_mean",

        "fold":
            "1-5"
    }

    mean_row.update(
        mean_metrics
    )

    # ========================================================
    # MEDIAN ENSEMBLE
    #
    # Median probability of five fold models
    # before thresholding.
    # ========================================================

    median_probs = np.median(
        prediction_matrix,
        axis=0
    )

    median_metrics = (
        evaluate_probabilities(
            median_probs,
            labels
        )
    )

    median_row = {

        "ablation":
            ablation_name,

        "model":
            "ENSEMBLE_MEDIAN",

        "type":
            "ensemble_median",

        "fold":
            "1-5"
    }

    median_row.update(
        median_metrics
    )

    # ========================================================
    # Combine
    # ========================================================

    summary_df = pd.DataFrame(
        [
            avg_row,
            std_row
        ]
    )

    ensemble_df = pd.DataFrame(
        [
            mean_row,
            median_row
        ]
    )

    fold_and_summary_df = pd.concat(
        [
            fold_df,
            summary_df
        ],
        ignore_index=True
    )

    final_group_df = pd.concat(
        [
            fold_and_summary_df,
            ensemble_df
        ],
        ignore_index=True
    )

    # ========================================================
    # Print ensemble result
    # ========================================================

    print()
    print(
        "=" * 80
    )

    print(
        f"ENSEMBLE RESULTS: "
        f"{ablation_name}"
    )

    print(
        "=" * 80
    )

    for name, metrics in [

        (
            "MEAN",
            mean_metrics
        ),

        (
            "MEDIAN",
            median_metrics
        )
    ]:

        print()
        print(
            f"{name} ENSEMBLE"
        )

        print(
            f"ROC-AUC = "
            f"{metrics['ROC_AUC']:.4f}"
        )

        print(
            f"PR-AUC  = "
            f"{metrics['PR_AUC']:.4f}"
        )

        for threshold in FIXED_THRESHOLDS:

            ts = str(
                threshold
            ).replace(
                ".",
                "_"
            )

            print(
                f"Threshold {threshold}: "
                f"MCC="
                f"{metrics[f't{ts}_MCC']:.4f}, "
                f"Precision="
                f"{metrics[f't{ts}_Precision']:.4f}, "
                f"Recall="
                f"{metrics[f't{ts}_Recall']:.4f}, "
                f"F1="
                f"{metrics[f't{ts}_F1']:.4f}, "
                f"Accuracy="
                f"{metrics[f't{ts}_Accuracy']:.4f}"
            )

    return final_group_df


# ============================================================
# Main
# ============================================================

def main(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    print()

    # ========================================================
    # Collect and GROUP all 80 checkpoints
    # ========================================================

    groups = collect_checkpoints(
        args.model_dir,
        device
    )

    if len(groups) == 0:

        raise RuntimeError(
            "No fold checkpoints found."
        )

    # ========================================================
    # Evaluate each ablation separately
    # ========================================================

    all_results = []

    for ablation_name in sorted(
        groups.keys()
    ):

        group_results = (
            evaluate_ablation_group(
                ablation_name,
                groups[ablation_name],
                args,
                device
            )
        )

        all_results.append(
            group_results
        )

    # ========================================================
    # Combine all ablations
    # ========================================================

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    # ========================================================
    # Save complete result
    # ========================================================

    final_df.to_csv(
        args.out_csv,
        index=False
    )

    # ========================================================
    # Create compact ensemble table
    # ========================================================

    ensemble_df = final_df[
        final_df["type"].isin(
            [
                "ensemble_mean",
                "ensemble_median"
            ]
        )
    ].copy()

    ensemble_csv = os.path.splitext(
        args.out_csv
    )[0] + "_ENSEMBLES.csv"

    ensemble_df.to_csv(
        ensemble_csv,
        index=False
    )

    # ========================================================
    # Create fold average table
    # ========================================================

    fold_average_df = final_df[
        final_df["type"].isin(
            [
                "fold_average",
                "fold_std"
            ]
        )
    ].copy()

    fold_average_csv = os.path.splitext(
        args.out_csv
    )[0] + "_FOLD_AVG_STD.csv"

    fold_average_df.to_csv(
        fold_average_csv,
        index=False
    )

    # ========================================================
    # Print final information
    # ========================================================

    print()
    print("=" * 90)
    print("ALL EVALUATIONS COMPLETED")
    print("=" * 90)

    print(
        f"Number of ablation groups: "
        f"{len(groups)}"
    )

    print(
        f"Total fold checkpoints used: "
        f"{sum(len(v) for v in groups.values())}"
    )

    print()
    print(
        "Complete results:"
    )

    print(
        args.out_csv
    )

    print()
    print(
        "Ensemble-only results:"
    )

    print(
        ensemble_csv
    )

    print()
    print(
        "Fold average/std results:"
    )

    print(
        fold_average_csv
    )

    print()
    print("=" * 90)

    # --------------------------------------------------------
    # Print compact table
    # --------------------------------------------------------

    display_columns = [
        "ablation",
        "type",
        "ROC_AUC",
        "PR_AUC",
        "t0_4_MCC",
        "t0_5_MCC",
        "t0_55_MCC",
        "t0_6_MCC",
        "t0_4_F1",
        "t0_5_F1",
        "t0_55_F1",
        "t0_6_F1"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in ensemble_df.columns
    ]

    print(
        ensemble_df[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json_folder",
        required=True
    )

    parser.add_argument(
        "--rnabert_dir",
        default=None
    )

    parser.add_argument(
        "--ernie_dir",
        default=None
    )

    parser.add_argument(
        "--rnafm_dir",
        default=None
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        help=(
            "Folder containing all CV checkpoints. "
            "The script automatically groups them "
            "by ablation_name."
        )
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--out_csv",
        default="cv_ensemble_all_results.csv"
    )

    args = parser.parse_args()

    main(args)