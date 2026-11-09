import torch
import numpy as np
from multimolecule import RnaTokenizer, RnaErnieModel

class ERNIEEmbedder:
    def __init__(self, device="cpu"):
        self.device = device

        self.tokenizer = RnaTokenizer.from_pretrained(
            "multimolecule/rnaernie"
        )
        self.model = RnaErnieModel.from_pretrained(
            "multimolecule/rnaernie"
        ).to(device)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed_sequence(self, seq):

        inputs = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state.squeeze(0)

        # Remove <cls> and <sep>
        token_embeds = hidden[1:-1]

        return token_embeds.cpu().numpy()
