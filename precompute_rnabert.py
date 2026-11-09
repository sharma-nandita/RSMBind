import os, json, numpy as np
from rnabert_embedder import RNABertEmbedder
import torch

# JSON_DIR = "E:\RNA_binding_desktop\chatgpt_multi_ligand_approach_ver3\data1\whole_train"
# OUT_DIR  = "rnabert_embeddings"
# JSON_DIR = "E:/RNA_binding_desktop/chatgpt_multi_ligand_approach_ver3/data1/te18_test"
# OUT_DIR  = "rnabert_embeddings_te18_test"
# JSON_DIR = "E:/RNA_binding_desktop/chatgpt_multi_ligand_approach_ver3/data1/tr60_train"
# OUT_DIR  = "rnabert_embeddings_tr60_train"
# JSON_DIR = "E:/RNA_binding_desktop/chatgpt_multi_ligand_approach_ver3/data/tr60_train"
# OUT_DIR  = "rnabert_embeddings_tr60_train_default_sasa"

# JSON_DIR = "E:/RNA_binding_desktop/chatgpt_multi_ligand_approach_ver3/data/te18_test"
# OUT_DIR  = "rnabert_embeddings_te18_test_def_sasa"

# JSON_DIR = "E:/RNA_binding_desktop/chatgpt_multi_ligand_approach_ver3/data1/rb9_test"
# OUT_DIR  = "rnabert_embeddings_rb9_test"
JSON_DIR = "E:/RNA_binding_desktop/dataset_design/train_json_438_len"
OUT_DIR =  "rnabert_embeddings_Hariboss_TR60"

os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = RNABertEmbedder(device=device)

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

# import os, json, numpy as np
# from rnabert_embedder import RNABertEmbedder
# import torch

# JSON_DIR = "E:\RNA_binding_desktop\chatgpt_multi_ligand_approach_ver3\data1\whole_train"
# OUT_DIR  = "rnabert_embeddings"

# os.makedirs(OUT_DIR, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedder = RNABertEmbedder(device=device)

# for fname in os.listdir(JSON_DIR):
#     if not fname.endswith(".json"):
#         continue

#     path = os.path.join(JSON_DIR, fname)
#     with open(path) as f:
#         data = json.load(f)

#     nodes = data["rna"]["nodes"]
#     seq = "".join([n["base"] for n in nodes])

#     emb = embedder.embed_sequence(seq)

#     assert emb.shape[0] == len(nodes), "Length mismatch!"

#     np.save(
#         os.path.join(OUT_DIR, fname.replace(".json", ".npy")),
#         emb
#     )

#     print("Saved:", fname)
