import os, json, numpy as np
from ERNIE_embedder import ERNIEEmbedder
import torch


# JSON_DIR = "E:/RNA_binding_desktop/binidnig_new_addition/TR60_train_json"
# OUT_DIR  = "ernie_embeddings_tr60_train"
# JSON_DIR = "E:/RNA_binding_desktop/binidnig_new_addition/rb9_json"
# OUT_DIR =  "ernie_embeddings_rb9"

# JSON_DIR = "E:/RNA_binding_desktop/binidnig_new_addition/TE18_json"
# OUT_DIR =  "ernie_embeddings_TE18"

# JSON_DIR = "E:/RNA_binding_desktop/binidnig_new_addition/JL10_json"
# OUT_DIR =  "ernie_embeddings_JL10"

# JSON_DIR = "E:/RNA_binding_desktop/binidnig_new_addition/TL12_json"
# OUT_DIR =  "ernie_embeddings_TL12"

JSON_DIR = "E:/RNA_binding_desktop/dataset_design/train_json_438_len"
OUT_DIR =  "E:/RNA_binding_desktop/dataset_design/ernie_embeddings_haribos_train"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = ERNIEEmbedder(device=device)

for fname in os.listdir(JSON_DIR):
    if not fname.endswith(".json"):
        continue

    path = os.path.join(JSON_DIR, fname)
    with open(path) as f:
        data = json.load(f)

    nodes = data["rna"]["nodes"]
    seq = "".join([n["base"] for n in nodes])

    # 🔥 RNABERT embedding
    emb = embedder.embed_sequence(seq)

    # ✅ SANITY CHECK (CRITICAL)
    assert emb.shape[0] == len(nodes), \
        f"{fname}: RNABERT len {emb.shape[0]} vs nodes {len(nodes)}"

    # 💾 Save
    np.save(
        os.path.join(OUT_DIR, fname.replace(".json", ".npy")),
        emb
    )

    print("Saved:", fname)

