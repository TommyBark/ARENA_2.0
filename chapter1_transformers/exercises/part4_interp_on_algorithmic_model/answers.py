# %%
import os

os.environ["ACCELERATE_DISABLE_RICH"] = "1"
import sys
from functools import partial
import json
from typing import List, Tuple, Union, Optional, Callable, Dict
import torch as t
from torch import Tensor
from sklearn.linear_model import LinearRegression
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import einops
from tqdm import tqdm
from jaxtyping import Float, Int, Bool
from pathlib import Path
import pandas as pd
import circuitsvis as cv
import webbrowser
from IPython.display import display
from transformer_lens import (
    utils,
    ActivationCache,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part4_interp_on_algorithmic_model"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import plotly_utils
from plotly_utils import hist, bar, imshow
import part4_interp_on_algorithmic_model.tests as tests
from part4_interp_on_algorithmic_model.brackets_datasets import (
    SimpleTokenizer,
    BracketsDataset,
)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
VOCAB = "()"

cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional",  # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB) + 3,  # plus 3 because of end and pad and start token
    d_vocab_out=2,  # 2 because we're doing binary classification
    use_attn_result=True,
    device=device,
    use_hook_tokens=True,
)

model = HookedTransformer(cfg).eval()

state_dict = t.load(section_dir / "brackets_model_state_dict.pt", map_location=device)
model.load_state_dict(state_dict)

# %%
tokenizer = SimpleTokenizer("()")

# Examples of tokenization
# (the second one applies padding, since the sequences are of different lengths)
print(tokenizer.tokenize("()"))
print(tokenizer.tokenize(["()", "()()"]))

# Dictionaries mapping indices to tokens and vice versa
print(tokenizer.i_to_t)
print(tokenizer.t_to_i)

# Examples of decoding (all padding tokens are removed)
print(tokenizer.decode(t.tensor([[0, 3, 4, 2, 1, 1]])))


# %%
def add_perma_hooks_to_mask_pad_tokens(
    model: HookedTransformer, pad_token: int
) -> HookedTransformer:
    # Hook which operates on the tokens, and stores a mask where tokens equal [pad]
    def cache_padding_tokens_mask(
        tokens: Float[Tensor, "batch seq"], hook: HookPoint
    ) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(
            tokens == pad_token, "b sK -> b 1 1 sK"
        )

    # Apply masking, by referencing the mask stored in the `hook_tokens` hook context
    def apply_padding_tokens_mask(
        attn_scores: Float[Tensor, "batch head seq_Q seq_K"],
        hook: HookPoint,
    ) -> None:
        attn_scores.masked_fill_(
            model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5
        )
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    # Add these hooks as permanent hooks (i.e. they aren't removed after functions like run_with_hooks)
    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_padding_tokens_mask)

    return model


model.reset_hooks(including_permanent=True)
model = add_perma_hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)
# %%
N_SAMPLES = 5000
with open(section_dir / "brackets_data.json") as f:
    data_tuples: List[Tuple[str, bool]] = json.load(f)
    print(f"loaded {len(data_tuples)} examples")
assert isinstance(data_tuples, list)
data_tuples = data_tuples[:N_SAMPLES]
data = BracketsDataset(data_tuples).to(device)
data_mini = BracketsDataset(data_tuples[:100]).to(device)
# %%
hist(
    [len(x) for x, _ in data_tuples],
    nbins=data.seq_length,
    title="Sequence lengths of brackets in dataset",
    labels={"x": "Seq len"},
)

# %%
# Define and tokenize examples
examples = [
    "()()",
    "(())",
    "))((",
    "()",
    "((()()()()))",
    "(()()()(()(())()",
    "()(()(((())())()))",
]
labels = [True, True, False, True, True, False, True]
toks = tokenizer.tokenize(examples)

# Get output logits for the 0th sequence position (i.e. the [start] token)
logits = model(toks)[:, 0]

# Get the probabilities via softmax, then get the balanced probability (which is the second element)
prob_balanced = logits.softmax(-1)[:, 1]

# Display output
print(
    "Model confidence:\n"
    + "\n".join(
        [
            f"{ex:18} : {prob:<8.4%} : label={int(label)}"
            for ex, prob, label in zip(examples, prob_balanced, labels)
        ]
    )
)


# %%
def run_model_on_data(
    model: HookedTransformer, data: BracketsDataset, batch_size: int = 200
) -> Float[Tensor, "batch 2"]:
    """Return probability that each example is balanced"""
    all_logits = []
    for i in tqdm(range(0, len(data.strs), batch_size)):
        toks = data.toks[i : i + batch_size]
        logits = model(toks)[:, 0]
        all_logits.append(logits)
    all_logits = t.cat(all_logits)
    assert all_logits.shape == (len(data), 2)
    return all_logits


test_set = data
n_correct = (
    run_model_on_data(model, test_set).argmax(-1).bool() == test_set.isbal
).sum()
print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
# %%


def bracket_classifier(s: str) -> bool:
    balance = 0
    for c in s:
        if balance < 0:
            return False
        if c == "(":
            balance += 1
        if c == ")":
            balance -= 1
    return balance == 0


for parens, expected in zip(examples, labels):
    actual = bracket_classifier(parens)
    assert expected == actual, f"{parens}: expected {expected} got {actual}"
print("is_balanced_forloop ok!")


# %%
def is_balanced_vectorized(tokens: Float[Tensor, "seq_len"]) -> bool:
    """
    Return True if the parens are balanced.

    tokens is a vector which has start/pad/end indices (0/1/2) as well as left/right brackets (3/4)
    """

    table = t.tensor([0, 0, 0, 1, -1])
    stripped = table[tokens]

    cum = t.cumsum(stripped, -1)
    if cum.min() < 0:
        return False

    return cum[-1] == 0


is_balanced_vectorized(t.tensor([1, 2, 3, 4, 3, 4, 0, 1]))

# %%
for tokens, expected in zip(tokenizer.tokenize(examples), labels):
    actual = is_balanced_vectorized(tokens)
    assert expected == actual, f"{tokens}: expected {expected} got {actual}"
print("is_balanced_vectorized ok!")


# %%
def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
    """
    return model.W_U[:, 0] - model.W_U[:, 1]


tests.test_get_post_final_ln_dir(get_post_final_ln_dir, model)


# %%
def get_activations(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    names: Union[str, List[str]],
) -> Union[t.Tensor, ActivationCache]:
    """
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    """
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache


def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    """
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.ln_final) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    """
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name(
            "resid_pre" if ln == "ln1" else "resid_mid", layer
        )
        output_hook_name = utils.get_act_name("normalized", layer, ln)

    return input_hook_name, output_hook_name


pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
print(pre_final_ln_name, post_final_ln_name)


# %%
def get_ln_fit(
    model: HookedTransformer,
    data: BracketsDataset,
    layernorm: LayerNorm,
    seq_pos: Optional[int] = None,
) -> Tuple[LinearRegression, float]:
    """
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
    """
    input_hook_name, output_hook_name = LN_hook_names(layernorm)
    input_act = get_activations(model, data.toks, input_hook_name)
    output_act = get_activations(model, data.toks, output_hook_name)
    if seq_pos is None:
        input = einops.rearrange(
            input_act, "batch seq_pos d_model -> (batch seq_pos) d_model"
        )
        output = einops.rearrange(
            output_act, "batch seq_pos d_model -> (batch seq_pos) d_model"
        )
    else:
        input = input_act[:, seq_pos, :]
        output = output_act[:, seq_pos, :]
    lin_model = LinearRegression()
    lin_model.fit(input, output)
    return lin_model, lin_model.score(input, output)


# L_final - (seq_len, seq_len)

# tests.test_get_ln_fit(get_ln_fit, model, data_mini)

(final_ln_fit_1, r2) = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)
print(f"r^2 for LN_final, at sequence position 0: {r2:.4f}")

(final_ln_fit_2, r2) = get_ln_fit(
    model, data, layernorm=model.blocks[1].ln1, seq_pos=None
)
print(f"r^2 for LN1, layer 1, over all sequence positions: {r2:.4f}")


# %%
def get_pre_final_ln_dir(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    """
    return t.from_numpy(
        get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)[0].coef_.T
    ) @ get_post_final_ln_dir(model)


tests.test_get_pre_final_ln_dir(get_pre_final_ln_dir, model, data_mini)
# %%


def get_out_by_components(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "component batch seq_pos emb"]:
    """
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2].
    The embeddings are the sum of token and positional embeddings.
    """
    # SOLUTION
    embedding_hook_names = ["hook_embed", "hook_pos_embed"]
    head_hook_names = [
        utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)
    ]
    mlp_hook_names = [
        utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)
    ]

    all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
    activations = get_activations(model, data.toks, all_hook_names)

    out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

    for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
        out = t.concat(
            [
                out,
                einops.rearrange(
                    activations[head_hook_name],
                    "batch seq heads emb -> heads batch seq emb",
                ),
                activations[mlp_hook_name].unsqueeze(0),
            ]
        )

    return out


# %%
biases = model.b_O.sum(0)
out_by_components = get_out_by_components(model, data)
summed_terms = out_by_components.sum(dim=0) + biases

final_ln_input_name, final_ln_output_name = LN_hook_names(model.ln_final)
final_ln_input = get_activations(model, data.toks, final_ln_input_name)

t.testing.assert_close(summed_terms, final_ln_input)
print("Tests passed!")
# %%

out_components = out_by_components[:, :, 0, :]
pre_final_ln_dir = get_pre_final_ln_dir(model, data)
out_by_component_in_unbalanced_dir = einops.einsum(
    out_components,
    pre_final_ln_dir,
    "component batch d_model, d_model -> component batch",
)

mean_balanced_components = einops.einsum(
    out_components,
    pre_final_ln_dir,
    "component batch d_model, d_model -> component batch",
)[:, data.isbal].mean(axis=1, keepdim=True)

out_by_component_in_unbalanced_dir -= mean_balanced_components
tests.test_out_by_component_in_unbalanced_dir(
    out_by_component_in_unbalanced_dir, model, data
)

plotly_utils.hists_per_comp(
    out_by_component_in_unbalanced_dir, data, xaxis_range=[-10, 20]
)


# %%
def is_balanced_vectorized_return_both(
    toks: Float[Tensor, "batch seq"]
) -> Tuple[Bool[Tensor, "batch"], Bool[Tensor, "batch"]]:
    # SOLUTION
    table = t.tensor([0, 0, 0, 1, -1]).to(device)
    change = table[toks.to(device)].flip(-1)
    altitude = t.cumsum(change, -1)
    total_elevation_failure = altitude[:, -1] != 0
    negative_failure = altitude.max(-1).values > 0
    return total_elevation_failure, negative_failure


total_elevation_failure, negative_failure = is_balanced_vectorized_return_both(
    data.toks
)

h20_in_unbalanced_dir = out_by_component_in_unbalanced_dir[7]
h21_in_unbalanced_dir = out_by_component_in_unbalanced_dir[8]

tests.test_total_elevation_and_negative_failures(
    data, total_elevation_failure, negative_failure
)
# %%
failure_types_dict = {
    "both failures": negative_failure & total_elevation_failure,
    "just neg failure": negative_failure & ~total_elevation_failure,
    "just total elevation failure": ~negative_failure & total_elevation_failure,
    "balanced": ~negative_failure & ~total_elevation_failure,
}

plotly_utils.plot_failure_types_scatter(
    h20_in_unbalanced_dir, h21_in_unbalanced_dir, failure_types_dict, data
)
# %%
plotly_utils.plot_contribution_vs_open_proportion(
    h21_in_unbalanced_dir,
    "Head 2.1 contribution vs proportion of open brackets '('",
    failure_types_dict,
    data,
)
# %%
plotly_utils.plot_contribution_vs_open_proportion(
    h20_in_unbalanced_dir,
    "Head 2.0 contribution vs proportion of open brackets '('",
    failure_types_dict,
    data,
)


# %%
def get_attn_probs(
    model: HookedTransformer, data: BracketsDataset, layer: int, head: int
) -> t.Tensor:
    """
    Returns: (N_SAMPLES, max_seq_len, max_seq_len) tensor that sums to 1 over the last dimension.
    """
    # tokens = tokenizer.tokenize(data)
    name = utils.get_act_name("pattern", layer)
    attention = get_activations(model, data.toks, names=name)
    return attention[:, head]


tests.test_get_attn_probs(get_attn_probs, model, data_mini)

# %%
attn_probs_20: Float[Tensor, "batch seqQ seqK"] = get_attn_probs(model, data, 2, 0)
attn_probs_20_open_query0 = attn_probs_20[data.starts_open].mean(0)[0]

bar(
    attn_probs_20_open_query0,
    title="Avg Attention Probabilities for query 0, first token '(', head 2.0",
    width=700,
    template="simple_white",
)


# %%
def get_pre_final_ln_dir(
    model: HookedTransformer, data: BracketsDataset
) -> Float[Tensor, "d_model"]:
    """
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    """
    return t.from_numpy(
        get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)[0].coef_.T
    ) @ get_post_final_ln_dir(model)


def get_WOV(
    model: HookedTransformer, layer: int, head: int
) -> Float[Tensor, "d_model d_model"]:
    """
    Returns the W_OV matrix for a particular layer and head.
    """
    W_O = model.W_O[layer, head]  # [layer, head]
    W_V = model.W_V[layer, head]  # [layer, head]
    # print(W_O.shape) # d_head, d_model
    # print(W_V.shape) # model, d_head
    return W_V @ W_O


def get_pre_20_dir(model, data) -> Float[Tensor, "d_model"]:
    """
    Returns the direction propagated back through the OV matrix of 2.0
    and then through the layernorm before the layer 2 attention heads.
    """
    pre_final_ln_dir = get_pre_final_ln_dir(model, data)  # vocab_out, d_head
    # print(pre_final_ln_dir.shape)
    W_OV20 = get_WOV(model, 2, 0)  # vocab_out, d_model
    L_1 = t.from_numpy(
        get_ln_fit(model, data, layernorm=model.blocks[2].ln1, seq_pos=1)[0].coef_.T
    )  # d_model , d_model
    # print(L_1.shape)
    return L_1 @ W_OV20 @ pre_final_ln_dir


tests.test_get_pre_20_dir(get_pre_20_dir, model, data_mini)
# %%
pre_layer2_outputs_seqpos1 = out_by_components[:-3, :, 1, :]
out_by_component_in_pre_20_unbalanced_dir = einops.einsum(
    pre_layer2_outputs_seqpos1,
    get_pre_20_dir(model, data),
    "comp batch emb, emb -> comp batch",
)
out_by_component_in_pre_20_unbalanced_dir -= out_by_component_in_pre_20_unbalanced_dir[
    :, data.isbal
].mean(-1, keepdim=True)


tests.test_out_by_component_in_pre_20_unbalanced_dir(
    out_by_component_in_pre_20_unbalanced_dir, model, data
)

plotly_utils.hists_per_comp(
    out_by_component_in_pre_20_unbalanced_dir, data, xaxis_range=(-5, 12)
)
# %%
plotly_utils.mlp_attribution_scatter(
    out_by_component_in_pre_20_unbalanced_dir, data, failure_types_dict
)


# %%
def get_out_by_neuron(
    model: HookedTransformer,
    data: BracketsDataset,
    layer: int,
    seq: Optional[int] = None,
) -> Float[Tensor, "batch *seq neuron d_model"]:
    """
    If seq is not None, then out[b, s, i, :] = f(x[b, s].T @ W_in[:, i]) @ W_out[i, :],
    i.e. the vector which is written to the residual stream by the ith neuron (where x
    is the input to the residual stream (i.e. shape (batch, seq, d_model)).

    If seq is None, then out[b, i, :] = vector f(x[b].T @ W_in[:, i]) @ W_out[i, :]

    (Note, using * in jaxtyping indicates an optional dimension)
    """
    name = utils.get_act_name("post", layer)

    # If seq is not None, then out[b, s, i, :] = f(x[b, s].T @ W_in[:, i]) @ W_out[i, :],
    if seq is not None:
        x = get_activations(model, data.toks, name)[:, seq, :]
        # print(x)
        # W_in = model.W_in[layer]
        W_out = model.W_out[layer]
        return einops.einsum(
            x, W_out, "batch d_hidden, d_hidden d_model -> batch d_hidden d_model"
        )
    else:
        x = get_activations(model, data.toks, name)
        W_out = model.W_out[layer]
        return einops.einsum(
            x,
            W_out,
            "batch seq d_hidden, d_hidden d_model -> batch seq d_hidden d_model",
        )


def get_out_by_neuron_in_20_dir(
    model: HookedTransformer, data: BracketsDataset, layer: int
) -> Float[Tensor, "batch neurons"]:
    """
    [b, s, i]th element is the contribution of the vector written by the ith neuron to the residual stream in the
    unbalanced direction (for the b-th element in the batch, and the s-th sequence position).

    In other words we need to take the vector produced by the `get_out_by_neuron` function, and project it onto the
    unbalanced direction for head 2.0 (at seq pos = 1).
    """
    out_neuron = get_out_by_neuron(model, data, layer, seq=1)  # batch, dhidden, dmodel
    dir20 = get_pre_20_dir(model, data)  # dmodel
    return einops.einsum(
        out_neuron, dir20, "batch dhidden dmodel, dmodel -> batch dhidden"
    )


tests.test_get_out_by_neuron(get_out_by_neuron, model, data_mini)
tests.test_get_out_by_neuron_in_20_dir(get_out_by_neuron_in_20_dir, model, data_mini)


# %%
def get_out_by_neuron_in_20_dir_less_memory(
    model: HookedTransformer, data: BracketsDataset, layer: int
) -> Float[Tensor, "batch neurons"]:
    """
    Has the same output as `get_out_by_neuron_in_20_dir`, but uses less memory (because it never stores
    the output vector of each neuron individually).
    """
    name = utils.get_act_name("post", layer)
    x = get_activations(model, data.toks, name)[:, 1, :]  # batch, seq, dhidden
    dir20 = get_pre_20_dir(model, data)  # dmodel
    W_out = model.W_out[layer]
    W_out_dir = W_out @ dir20
    return einops.einsum(x, W_out_dir, "batch dhidden, dhidden -> batch dhidden")
    # return x * W_out_dir


tests.test_get_out_by_neuron_in_20_dir_less_memory(
    get_out_by_neuron_in_20_dir_less_memory, model, data_mini
)


# %%
for layer in range(2):
    # Get neuron significances for head 2.0, sequence position #1 output
    neurons_in_unbalanced_dir = get_out_by_neuron_in_20_dir_less_memory(
        model, data, layer
    )[utils.to_numpy(data.starts_open), :]
    # Plot neurons' activations
    plotly_utils.plot_neurons(
        neurons_in_unbalanced_dir,
        model,
        data,
        failure_types_dict,
        layer,
        renderer="browser",
    )


# %%
def get_q_and_k_for_given_input(
    model: HookedTransformer,
    tokenizer: SimpleTokenizer,
    parens: str,
    layer: int,
) -> Tuple[Float[Tensor, "seq_d_model"], Float[Tensor, "seq_d_model"]]:
    """
    Returns the queries and keys (both of shape [seq, d_model]) for the given parns input, in the attention head `layer.head`.
    """
    q_name = utils.get_act_name("q", layer)
    k_name = utils.get_act_name("k", layer)
    q = get_activations(
        model, tokenizer.tokenize(parens), q_name
    )  # batch, seq, n_head, d_head
    k = get_activations(
        model, tokenizer.tokenize(parens), k_name
    )  # batch, seq, n_head, d_head
    return q[0], k[0]


tests.test_get_q_and_k_for_given_input(get_q_and_k_for_given_input, model, tokenizer)

# %%
layer = 0
all_left_parens = "".join(["(" * 40])
all_right_parens = "".join([")" * 40])

model.reset_hooks()
q0_all_left, k0_all_left = get_q_and_k_for_given_input(
    model, tokenizer, all_left_parens, layer
)
q0_all_right, k0_all_right = get_q_and_k_for_given_input(
    model, tokenizer, all_right_parens, layer
)
k0_avg = (k0_all_left + k0_all_right) / 2


# Define hook function to patch in q or k vectors
def hook_fn_patch_qk(
    value: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    new_value: Float[Tensor, "... seq d_head"],
    head_idx: Optional[int] = None,
) -> None:
    if head_idx is not None:
        value[..., head_idx, :] = new_value[..., head_idx, :]
    else:
        value[...] = new_value[...]


# Define hook function to display attention patterns (using plotly)
def hook_fn_display_attn_patterns(
    pattern: Float[Tensor, "batch heads seqQ seqK"], hook: HookPoint, head_idx: int = 0
) -> None:
    avg_head_attn_pattern = pattern.mean(0)
    labels = ["[start]", *[f"{i+1}" for i in range(40)], "[end]"]
    display(
        cv.attention.attention_heads(
            tokens=labels,
            attention=avg_head_attn_pattern,
            attention_head_names=["0.0", "0.1"],
            max_value=avg_head_attn_pattern.max(),
        )
    )


# Run our model on left parens, but patch in the average key values for left vs right parens
# This is to give us a rough idea how the model behaves on average when the query is a left paren
model.run_with_hooks(
    tokenizer.tokenize(all_left_parens).to(device),
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("k", layer), partial(hook_fn_patch_qk, new_value=k0_avg)),
        (utils.get_act_name("pattern", layer), hook_fn_display_attn_patterns),
    ],
)


# %%
def hook_fn_display_attn_patterns_for_single_query(
    pattern: Float[Tensor, "batch heads seqQ seqK"],
    hook: HookPoint,
    head_idx: int = 0,
    query_idx: int = 1,
):
    bar(
        utils.to_numpy(pattern[:, head_idx, query_idx].mean(0)),
        title=f"Average attn probabilities on data at posn 1, with query token = '('",
        labels={
            "index": "Sequence position of key",
            "value": "Average attn over dataset",
        },
        height=500,
        width=800,
        yaxis_range=[0, 0.1],
        template="simple_white",
    )


data_len_40 = BracketsDataset.with_length(data_tuples, 40).to(device)

model.reset_hooks()
model.run_with_hooks(
    data_len_40.toks[data_len_40.isbal],
    return_type=None,
    fwd_hooks=[
        (utils.get_act_name("q", 0), partial(hook_fn_patch_qk, new_value=q0_all_left)),
        (
            utils.get_act_name("pattern", 0),
            hook_fn_display_attn_patterns_for_single_query,
        ),
    ],
)


# %%
def embedding(
    model: HookedTransformer, tokenizer: SimpleTokenizer, char: str
) -> Float[Tensor, "d_model"]:
    assert char in ("(", ")")
    idx = tokenizer.t_to_i[char]
    return model.W_E[idx]


L = get_ln_fit(model, data, layernorm=model.blocks[0].ln1, seq_pos=1)[0].coef_
v_L = embedding(model, tokenizer, "(").T @ L.T @ get_WOV(model, 0, 0)
v_R = embedding(model, tokenizer, ")").T @ L.T @ get_WOV(model, 0, 0)
print("Cosine similarity: ", t.cosine_similarity(v_L, v_R, dim=0).item())


# %%
def cos_sim_with_MLP_weights(
    model: HookedTransformer, v: Float[Tensor, "d_model"], layer: int
) -> Float[Tensor, "d_mlp"]:
    """
    Returns a vector of length d_mlp, where the ith element is the cosine similarity between v and the
    ith in-direction of the MLP in layer `layer`.

    Recall that the in-direction of the MLPs are the columns of the W_in matrix.
    """
    w = model.W_in[layer].T  # dmodel, dhidden # 56 56
    return t.cosine_similarity(v, w, dim=1)  # should be dim=0


def avg_squared_cos_sim(v: Float[Tensor, "d_model"], n_samples: int = 1000) -> float:
    """
    Returns the average (over n_samples) cosine similarity between v and another randomly chosen vector.

    We can create random vectors from the standard N(0, I) distribution.
    """
    random_vectors = t.randn(size=(n_samples, v.shape[0]))
    return t.cosine_similarity(v, random_vectors, dim=1).pow(2).mean()  # dim=1


print("Avg squared cosine similarity of v_R with ...\n")

cos_sim_mlp0 = cos_sim_with_MLP_weights(model, v_R, 0)
print(f"...MLP input directions in layer 0:  {cos_sim_mlp0.pow(2).mean():.6f}")

cos_sim_mlp1 = cos_sim_with_MLP_weights(model, v_R, 1)
print(f"...MLP input directions in layer 1:  {cos_sim_mlp1.pow(2).mean():.6f}")

cos_sim_rand = avg_squared_cos_sim(v_R)
print(f"...random vectors of len = d_model:  {cos_sim_rand:.6f}")

# %%
