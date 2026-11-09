import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GATv2Conv
# from torch_geometric.nn import GraphConv
# from torch_geometric.nn import RGCNConv
from torch_geometric.nn import GATv2Conv


class RNAOnlyGNN(nn.Module):

    HANDCRAFTED_DIM = 9
    RNABERT_DIM = 120
    ERNIE_DIM = 768
    RNAFM_DIM = 640

    RNABERT_PROJ = 32
    ERNIE_PROJ = 128
    RNAFM_PROJ = 128

    # RNABERT_PROJ = 64
    # ERNIE_PROJ = 256
    # RNAFM_PROJ = 256

    def __init__(
        self,
        in_node_dim,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
        use_rnabert=False,
        use_ernie=False,
        use_rnafm=False,
    ):

        super().__init__()

        self.use_rnabert = use_rnabert
        self.use_ernie = use_ernie
        self.use_rnafm = use_rnafm

        # --------------------------------------------------
        # RNABERT Projection
        # --------------------------------------------------
        if self.use_rnabert:
            self.rnabert_proj = nn.Sequential(
                nn.LayerNorm(120),
                nn.Linear(120, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU()
            )

        # --------------------------------------------------
        # ERNIE Projection
        # --------------------------------------------------
        if self.use_ernie:
            self.ernie_proj = nn.Sequential(
                nn.LayerNorm(768),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),

                nn.Linear(256, 128),
                nn.ReLU()
                # nn.Dropout(0.1),

                # nn.Linear(256, 128),
                # nn.ReLU()
            )

        # --------------------------------------------------
        # RNAFM Projection
        # --------------------------------------------------
        if self.use_rnafm:
            self.rnafm_proj = nn.Sequential(
                nn.LayerNorm(640),
                nn.Linear(640, 512),
                nn.ReLU(),
                nn.Dropout(0.1),

                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),

                # nn.Linear(512, 256),
                # nn.ReLU(),
                # nn.Dropout(0.1),

                nn.Linear(256, 128),
                nn.ReLU()
            )
        
        # --------------------------------------------------
        # Input projection
        # --------------------------------------------------
        fusion_dim = self.HANDCRAFTED_DIM

        if self.use_rnabert:
            fusion_dim += self.RNABERT_PROJ

        if self.use_ernie:
            fusion_dim += self.ERNIE_PROJ

        if self.use_rnafm:
            fusion_dim += self.RNAFM_PROJ

        self.input_proj = nn.Linear(fusion_dim, hidden_dim)

        # --------------------------------------------------
        # GAT layers
        # --------------------------------------------------
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers - 1):

            self.convs.append(
                GATv2Conv(
                    hidden_dim,
                    hidden_dim // 4,
                    heads=4,
                    edge_dim=4,
                    dropout=0.1
                )
            )

            self.bns.append(nn.BatchNorm1d(hidden_dim))

        print("Number of GAT layers:", len(self.convs))

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        idx = 0

        features = []

        # -----------------------------------------
        # Handcrafted
        # -----------------------------------------
        handcrafted = x[:, idx:idx + self.HANDCRAFTED_DIM]
        features.append(handcrafted)
        idx += self.HANDCRAFTED_DIM

        # -----------------------------------------
        # RNABERT
        # -----------------------------------------
        if self.use_rnabert:
            rnabert = x[:, idx:idx + self.RNABERT_DIM]
            rnabert = self.rnabert_proj(rnabert)
            features.append(rnabert)
            idx += self.RNABERT_DIM

        # -----------------------------------------
        # ERNIE
        # -----------------------------------------
        if self.use_ernie:
            ernie = x[:, idx:idx + self.ERNIE_DIM]
            ernie = self.ernie_proj(ernie)
            features.append(ernie)
            idx += self.ERNIE_DIM

        # -----------------------------------------
        # RNAFM
        # -----------------------------------------
        if self.use_rnafm:
            rnafm = x[:, idx:idx + self.RNAFM_DIM]
            rnafm = self.rnafm_proj(rnafm)
            features.append(rnafm)
            idx += self.RNAFM_DIM

        x = torch.cat(features, dim=1)

        x = self.input_proj(x)


        # print("features shape:", len(features[0]), len(features[1]), len(features[2]))
        # print(features[0])
        # -----------------------------------------
        # GNN
        # -----------------------------------------
        for conv, bn in zip(self.convs, self.bns):

            residual = x

            x = conv(
                x,
                edge_index,
                edge_attr=edge_attr
            )

            x = bn(x)

            x = F.relu(x)

            x = F.dropout(
                x,
                p=0.2,
                training=self.training
            )

            x = x + residual

        logits = self.classifier(x).squeeze(-1)

        return logits