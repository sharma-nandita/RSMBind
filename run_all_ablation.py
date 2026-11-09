import os
import subprocess

TRAIN_SCRIPT = "train.py"

######## TR60 run ###############
# JSON_FOLDER = r"E:/RNA_binding_desktop/binidnig_new_addition/TR60_train_json"
# RNABERT_DIR = r"E:/RNA_binding_desktop/binidnig_new_addition/rnabert_embeddings_train"
# ERNIE_DIR   = r"E:/RNA_binding_desktop/binidnig_new_addition/ernie_embeddings_tr60_train"
# RNAFM_DIR   = r"E:/RNA_binding_desktop/binidnig_new_addition/rnafm_embeddings"

############ For hariboss dataset, 358 itms ###############
JSON_FOLDER = r"E:/RNA_binding_desktop\dataset_design/train_json_438_len"
RNABERT_DIR = r"E:/RNA_binding_desktop\dataset_design/rnabert_embeddings_Hariboss_TR60"
ERNIE_DIR   = r"E:/RNA_binding_desktop\dataset_design/ernie_embeddings_haribos_train"
RNAFM_DIR   = r"E:/RNA_binding_desktop\dataset_design/rnafm_embeddings_haribos_train"

OUTPUT_DIR = r"E:/RNA_binding_desktop/cv_ablation/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Embedding configurations
# =====================================================

embedding_configs = [

    ("hc", {}),

    ("hc_rnabert", {
        "rnabert": RNABERT_DIR
    }),

    ("hc_ernie", {
        "ernie": ERNIE_DIR
    }),

    ("hc_rnafm", {
        "rnafm": RNAFM_DIR
    }),

    ("hc_rnabert_ernie", {
        "rnabert": RNABERT_DIR,
        "ernie": ERNIE_DIR
    }),

    ("hc_rnabert_rnafm", {
        "rnabert": RNABERT_DIR,
        "rnafm": RNAFM_DIR
    }),

    ("hc_ernie_rnafm", {
        "ernie": ERNIE_DIR,
        "rnafm": RNAFM_DIR
    }),

    ("hc_all", {
        "rnabert": RNABERT_DIR,
        "ernie": ERNIE_DIR,
        "rnafm": RNAFM_DIR
    }),
]

# =====================================================
# Feature ablations
# =====================================================

feature_experiments = [

    ("full", []),

    # ("no_motif", ["--no_motif"]),

    ("no_sasa", ["--no_sasa"]),

    # ("no_geometry", ["--no_geometry"])

    # ("no_atomic", ["--no_atomic"]),

    # ("no_window", ["--no_window"]),
]

# =====================================================
# Run experiments
# =====================================================

total_runs = len(embedding_configs) * len(feature_experiments)
current_run = 1

for embed_name, embed_dirs in embedding_configs:

    for exp_name, flags in feature_experiments:


        #########
        experiment_name = f"{embed_name}_{exp_name}"

        # One completely separate directory for each CV experiment
        experiment_dir = os.path.join(
            OUTPUT_DIR,
            experiment_name
        )

        os.makedirs(experiment_dir, exist_ok=True)

        # Base checkpoint path passed to the k-fold training script
        save_path = os.path.join(
            experiment_dir,
            "model.pt"
        )

        cmd = [
            "python",
            TRAIN_SCRIPT,

            "--json_folder",
            JSON_FOLDER,

            "--save_path",
            save_path,

            "--ablation_name",
            experiment_name
        ]

        # ------------------------------------------
        # Optional embedding directories
        # ------------------------------------------

        if "rnabert" in embed_dirs:
            cmd.extend([
                "--rnabert_dir",
                embed_dirs["rnabert"]
            ])

        if "ernie" in embed_dirs:
            cmd.extend([
                "--ernie_dir",
                embed_dirs["ernie"]
            ])

        if "rnafm" in embed_dirs:
            cmd.extend([
                "--rnafm_dir",
                embed_dirs["rnafm"]
            ])

        # ------------------------------------------
        # Feature ablation flags
        # ------------------------------------------

        cmd.extend(flags)

        print("=" * 80)
        print(f"[{current_run}/{total_runs}] {experiment_name}")
        print("=" * 80)
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)

        current_run += 1

print("\n")
print("=" * 80)
print("All experiments completed successfully.")
print("=" * 80)