import torch
import numpy as np
from multimolecule import RnaTokenizer, RnaBertModel

class RNABertEmbedder:
    def __init__(self, device="cpu"):
        self.device = device

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/rnabert"
        )
        self.model = RnaBertModel.from_pretrained(
            "multimolecule/rnabert"
        ).to(device)

        self.model.eval()  # 🔒 frozen

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed_sequence(self, seq):
        """
        seq: string, e.g. 'AUGCGU...'
        returns: (L, hidden_dim) numpy array
        """

        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state.squeeze(0)

        # Remove special tokens: <cls> and <sep>
        token_embeds = hidden[1:-1]

        return token_embeds.cpu().numpy()
