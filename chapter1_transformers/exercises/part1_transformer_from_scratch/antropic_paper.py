# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import einops
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
import torch as t
from torch import Tensor
import torch.nn as nn
import datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb
from typing import Tuple, List, Optional, Dict
from jaxtyping import Float, Int
from solutions import Embed, PosEmbed, Unembed, LitTransformer

# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = False
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False
)


@dataclass
class TransformerTrainingArgs:
    batch_size = 8
    max_epochs = 1
    max_steps = 1000
    log_every = 10
    lr = 1e-3
    weight_decay = 1e-2
    log_dir: str = os.getcwd() + "/logs/day1/"
    log_name: str = "day1-transformer"
    run_name: Optional[str] = None
    log_every_n_steps: int = 1


args = TransformerTrainingArgs()


class Zero_Layer_Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        # self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        x = self.embed(tokens) + self.pos_embed(tokens)
        # x = self.ln_final(x)
        x = self.unembed(x)
        return x


model_zero = Zero_Layer_Transformer(cfg)
# %%
dataset = datasets.load_dataset("NeelNanda/pile-10k", split="train").remove_columns(
    "meta"
)
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model_zero.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=1,
)
data_loader = DataLoader(
    tokenized_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)
# %%
#litmodel_zero = LitTransformer(args, model_zero, data_loader)
litmodel_zero = LitTransformer.load_from_checkpoint("/home/ubuntu/model_zero_e1.ckpt",args=args,model=model_zero, data_loader = data_loader)
# %%
# logger = WandbLogger(save_dir=args.log_dir, project=args.log_name, name=args.run_name)

# # %%
# trainer = pl.Trainer(
#     max_epochs=args.max_epochs, logger=logger, log_every_n_steps=args.log_every_n_steps
# )
# trainer.fit(model=litmodel_zero, train_dataloaders=litmodel_zero.data_loader)
# wandb.finish()

# %%
# 1. Look if W_E is inverse of W_U


W_E = litmodel_zero.model.embed.W_E 
W_pos = litmodel_zero.model.pos_embed.W_pos
W_U = litmodel_zero.model.unembed.W_U
logits = (W_E) @ W_U

## TODO: look at those logits

# %% 
# 2. Observe if the model learned simple bigrams
test_str = "Vol"
test_tokens = t.tensor(reference_gpt2.tokenizer.encode(test_str)).unsqueeze(0)

with t.no_grad():
    for _ in range(10):
        ans2 = litmodel_zero(test_tokens)
        next_token = t.softmax(ans2[0,-1,:],dim=0).argmax().item()
        print(next_token)
        next_token_str = reference_gpt2.tokenizer.decode(next_token)
        test_str += next_token_str
        test_tokens = t.tensor(reference_gpt2.tokenizer.encode(test_str)).unsqueeze(0)

## RESULTS: cannot see, model is too dumb probably
# %%
