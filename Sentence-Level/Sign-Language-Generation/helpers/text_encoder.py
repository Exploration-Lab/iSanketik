"""
Text Encoder

Wraps a pretrained Flan-T5-base encoder to produce per-token text
embeddings for conditioning the DiT flow model. Supports partial or full
fine-tuning of the encoder's top transformer blocks, and provides a
matching null encoder that returns zero embeddings for the unconditional
pass used in classifier-free guidance.

Input:
  input_ids      : (B, L)  tokenized text
  attention_mask : (B, L)  1=valid 0=pad

Output:
  text_emb : (B, L, 768)  per-token embeddings
  attn_mask: (B, L)       same as input attention_mask, passed through for
                           use in downstream cross-attention
"""

import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer


class FlanT5TextEncoder(nn.Module):
    """
    Encodes English text into (B, L, text_dim) using the Flan-T5-base encoder.

    fine_tune_layers: number of top encoder layers to unfreeze.
                      0  = fully frozen (fastest, good baseline)
                      4  = fine-tune last 4 layers (recommended)
                      -1 = fine-tune entire encoder
    """

    MODEL_NAME = "google/flan-t5-base"
    HIDDEN_DIM = 768

    def __init__(
        self,
        fine_tune_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        print(f"Loading {self.MODEL_NAME} ...")
        self.encoder = T5EncoderModel.from_pretrained(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.hidden_dim = self.HIDDEN_DIM

        # Freeze all by default
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Optionally unfreeze top N transformer blocks
        if fine_tune_layers > 0:
            total_layers = len(self.encoder.encoder.block)
            unfreeze_from = total_layers - fine_tune_layers
            for i, block in enumerate(self.encoder.encoder.block):
                if i >= unfreeze_from:
                    for p in block.parameters():
                        p.requires_grad = True
            # Also unfreeze final layer norm
            for p in self.encoder.encoder.final_layer_norm.parameters():
                p.requires_grad = True
        elif fine_tune_layers == -1:
            for p in self.encoder.parameters():
                p.requires_grad = True

        self.dropout = nn.Dropout(dropout)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Text encoder trainable params: {n_trainable:,}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        input_ids      : (B, L)
        attention_mask : (B, L)
        Returns:
          text_emb : (B, L, 768), all token embeddings
          attn_mask: (B, L), same as input attention_mask (for cross-attn)
        """
        # Run the encoder in full precision regardless of the outer autocast
        # context, since T5 attention is numerically unstable under fp16.
        with torch.autocast(device_type='cuda', enabled=False):
            out = self.encoder(
                input_ids=input_ids.to(torch.long),
                attention_mask=attention_mask,
            )
        text_emb = self.dropout(out.last_hidden_state.float())
        return text_emb, attention_mask


class NullTextEncoder(nn.Module):
    """
    Returns zero embeddings, used for the classifier-free guidance
    unconditional pass. Matches the FlanT5TextEncoder interface.
    """
    def __init__(self, hidden_dim: int = 768, max_len: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        B, L = input_ids.shape
        zeros = torch.zeros(B, L, self.hidden_dim, device=input_ids.device)
        zero_mask = torch.zeros(B, L, device=input_ids.device)
        return zeros, zero_mask