# train_rna_only_v1_kfold.py
import argparse
import os
import copy
import torch
import numpy as np
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Batch
from torch.utils.data import Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    matthews_corrcoef
)


from dataset import RNALigandJSONDataset  #9 features, with edge
from GATv2Conv import RNAOnlyGNN

print("*******************************")


# -------------------------------------------------
# Reproducibility
# -------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------------------------------
# Training
# -------------------------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        logits = model(batch)

        loss = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------------------------------
# Evaluation (threshold sweep not fixed)
# -------------------------------------------------
@torch.no_grad()
def eval_model(model, loader, device):

    model.eval()

    probs_all = []
    labels_all = []

    for batch in loader:

        batch = batch.to(device)

        logits = model(batch)

        probs = torch.sigmoid(logits)

        probs_all.append(probs.cpu())
        labels_all.append(batch.y.cpu())

    probs_all = torch.cat(probs_all).numpy()
    labels_all = torch.cat(labels_all).numpy()

    auc = roc_auc_score(labels_all, probs_all)

    best_mcc = -1
    best_thr = 0.5

    for thr in np.arange(0.30, 0.81, 0.01):

        preds = (probs_all >= thr).astype(int)

        mcc = matthews_corrcoef(
            labels_all,
            preds
        )

        if mcc > best_mcc:

            best_mcc = mcc
            best_thr = thr

    preds = (probs_all >= best_thr).astype(int)

    f1 = f1_score(
        labels_all,
        preds,
        zero_division=0
    )

    acc = accuracy_score(
        labels_all,
        preds
    )

    return auc, f1, acc, best_mcc, best_thr


# -------------------------------------------------
# Helper: extract labels for stratification
# -------------------------------------------------
def get_labels(dataset):
    """
    Pulls a scalar label per graph so we can stratify the K-fold split.
    Assumes dataset[i].y is a single binary label (shape [1] or scalar).
    """
    labels = []
    for i in range(len(dataset)):
        y = dataset[i].y
        y = y.view(-1)[0].item()
        labels.append(int(round(y)))
    return np.array(labels)


# -------------------------------------------------
# Build a fresh model (so each fold starts from scratch)
# -------------------------------------------------
def build_model(args, input_dim, device):
    model = RNAOnlyGNN(
        in_node_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,

        use_rnabert=args.rnabert_dir is not None,
        use_ernie=args.ernie_dir is not None,
        use_rnafm=args.rnafm_dir is not None,
    ).to(device)
    return model


# -------------------------------------------------
# Train a single fold, return best metrics + save fold checkpoint
# -------------------------------------------------
def run_fold(fold_idx, train_ds, val_ds, input_dim, args, device):

    train_loader = GeoDataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: Batch.from_data_list(b)
    )

    val_loader = GeoDataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: Batch.from_data_list(b)
    )

    model = build_model(args, input_dim, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.pos_weight <= 0:
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        pos_weight = torch.tensor([args.pos_weight], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    fold_best_mcc = -1.0
    fold_best_state = None
    fold_best_metrics = None

    fold_save_path = os.path.join(
        args.save_dir,
        f"{args.ablation_name}_fold{fold_idx}_best.pt"
    )

    for epoch in range(1, args.epochs + 1):

        loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        auc, f1, acc, mcc, best_thr = eval_model(
            model, val_loader, device
        )

        print(
            f"[Fold {fold_idx}] Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"AUC {auc:.4f} | "
            f"F1 {f1:.4f} | "
            f"MCC {mcc:.4f} | "
            f"Thr {best_thr:.2f}"
        )

        if mcc > fold_best_mcc:
            fold_best_mcc = mcc
            fold_best_state = copy.deepcopy(model.state_dict())
            fold_best_metrics = {
                "auc": auc,
                "f1": f1,
                "acc": acc,
                "mcc": mcc,
                "thr": best_thr,
                "epoch": epoch,
            }

    # Save this fold's best checkpoint
    torch.save({
        "model_state": fold_best_state,

        "in_dim": input_dim,

        "use_rnabert": args.rnabert_dir is not None,
        "use_ernie": args.ernie_dir is not None,
        "use_rnafm": args.rnafm_dir is not None,

        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,

        "pos_weight": args.pos_weight,
        "batch_size": args.batch_size,

        "no_sasa": args.no_sasa,
        "no_geometry": args.no_geometry,

        "ablation_name": args.ablation_name,
        "fold": fold_idx,

        "best_val_mcc": fold_best_metrics["mcc"],
        "best_val_auc": fold_best_metrics["auc"],
        "best_val_f1": fold_best_metrics["f1"],
        "best_val_acc": fold_best_metrics["acc"],
        "best_val_thr": fold_best_metrics["thr"],
        "best_epoch": fold_best_metrics["epoch"],

    }, fold_save_path)

    print(
        f"[Fold {fold_idx}] Saved best model "
        f"(MCC={fold_best_metrics['mcc']:.4f}, "
        f"AUC={fold_best_metrics['auc']:.4f}) -> {fold_save_path}"
    )

    return fold_best_state, fold_best_metrics, fold_save_path


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):

    # set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("\n==============================")
    print("Experiment:", args.ablation_name)
    print(f"K-Fold CV: {args.n_splits} folds")
    print("==============================")

    os.makedirs(args.save_dir, exist_ok=True)

    # -------------------------------------------------
    # Dataset
    # -------------------------------------------------
    dataset = RNALigandJSONDataset(
        args.json_folder,
        rnabert_dir=args.rnabert_dir,
        ernie_dir=args.ernie_dir,
        rnafm_dir=args.rnafm_dir,

        use_sasa=not args.no_sasa,
        use_geometry=not args.no_geometry,
    )

    input_dim = dataset[0].x.shape[1]
    print("Node feature dimension:", input_dim)
    print(dataset[0])

    N = len(dataset)
    print("Total samples:", N)

    labels = get_labels(dataset)

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed
    )

    all_indices = np.arange(N)

    fold_results = []
    global_best_mcc = -1.0
    global_best_state = None
    global_best_fold = -1
    global_best_metrics = None

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(all_indices, labels), start=1
    ):

        print(f"\n---------- Fold {fold_idx}/{args.n_splits} ----------")
        print(f"Train size: {len(train_idx)} | Val size: {len(val_idx)}")

        train_ds = Subset(dataset, train_idx.tolist())
        val_ds = Subset(dataset, val_idx.tolist())

        fold_state, fold_metrics, fold_path = run_fold(
            fold_idx, train_ds, val_ds, input_dim, args, device
        )

        fold_results.append(fold_metrics)

        if fold_metrics["mcc"] > global_best_mcc:
            global_best_mcc = fold_metrics["mcc"]
            global_best_state = fold_state
            global_best_fold = fold_idx
            global_best_metrics = fold_metrics

    # -------------------------------------------------
    # Save overall best model (across all folds)
    # -------------------------------------------------
    torch.save({
        "model_state": global_best_state,

        "in_dim": input_dim,

        "use_rnabert": args.rnabert_dir is not None,
        "use_ernie": args.ernie_dir is not None,
        "use_rnafm": args.rnafm_dir is not None,

        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,

        "pos_weight": args.pos_weight,
        "batch_size": args.batch_size,

        "no_sasa": args.no_sasa,
        "no_geometry": args.no_geometry,

        "ablation_name": args.ablation_name,
        "best_fold": global_best_fold,
        "n_splits": args.n_splits,

        "best_val_mcc": global_best_metrics["mcc"],
        "best_val_auc": global_best_metrics["auc"],
        "best_val_f1": global_best_metrics["f1"],
        "best_val_acc": global_best_metrics["acc"],
        "best_val_thr": global_best_metrics["thr"],

    }, args.save_path)

    print("\n==============================")
    print("K-Fold CV complete.")
    print("==============================")

    aucs = [r["auc"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    accs = [r["acc"] for r in fold_results]
    mccs = [r["mcc"] for r in fold_results]

    for i, r in enumerate(fold_results, start=1):
        print(
            f"Fold {i}: AUC {r['auc']:.4f} | F1 {r['f1']:.4f} | "
            f"ACC {r['acc']:.4f} | MCC {r['mcc']:.4f} | Thr {r['thr']:.2f}"
        )

    print("-" * 60)
    print(f"AUC  mean={np.mean(aucs):.4f} std={np.std(aucs):.4f}")
    print(f"F1   mean={np.mean(f1s):.4f} std={np.std(f1s):.4f}")
    print(f"ACC  mean={np.mean(accs):.4f} std={np.std(accs):.4f}")
    print(f"MCC  mean={np.mean(mccs):.4f} std={np.std(mccs):.4f}")
    print("-" * 60)
    print(f"Best fold: {global_best_fold} (MCC={global_best_mcc:.4f})")
    print(f"Best overall model saved at: {args.save_path}")
    print(f"Per-fold models saved under: {args.save_dir}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--json_folder", type=str, required=True)
    parser.add_argument("--rnabert_dir", type=str, default=None)
    parser.add_argument("--ernie_dir", default=None)
    parser.add_argument("--rnafm_dir", default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)  # 128
    parser.add_argument("--num_layers", type=int, default=2)    # 2
    parser.add_argument("--dropout", type=float, default=0.0)   # 0.0

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # set <=0 to disable
    parser.add_argument("--pos_weight", type=float, default=1.5) # 1.5

    parser.add_argument("--no_sasa", action="store_true")
    parser.add_argument("--no_geometry", action="store_true")

    # K-fold CV controls
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of CV folds (e.g. 5 or 10)."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models_",
        help="Directory where each fold's best checkpoint is saved."
    )

    parser.add_argument(
        "--ablation_name",
        type=str,
        default="full"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="rna_only_best_mcc.pt",
        help="Path to save the single best model across all folds."
    )

    args = parser.parse_args()
    main(args)
