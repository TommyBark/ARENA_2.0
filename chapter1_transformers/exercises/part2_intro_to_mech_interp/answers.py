# %%
import os
import sys
import plotly.express as px
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import einops
from jaxtyping import Int, Float
from typing import List, Optional, Tuple
import functools
from tqdm import tqdm
from IPython.display import display
import webbrowser
import gdown
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
import circuitsvis as cv

# Make sure exercises are in the path
chapter = r"chapter1_transformers"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = (exercises_dir / "part2_intro_to_mech_interp").resolve()
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from plotly_utils import (
    imshow,
    hist,
    plot_comp_scores,
    plot_logit_attribution,
    plot_loss_difference,
)
from part1_transformer_from_scratch.solutions import get_log_probs
import part2_intro_to_mech_interp.tests as tests

# Saves computation time, since we don't need it for the contents of this notebook
t.set_grad_enabled(False)

device = t.device("cuda" if t.cuda.is_available() else "cpu")

MAIN = __name__ == "__main__"

# %%
gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
# %%
model_description_text = """## Loading Models

HookedTransformer comes loaded with >40 open source GPT-style models. You can load any of them in with `HookedTransformer.from_pretrained(MODEL_NAME)`. Each model is loaded into the consistent HookedTransformer architecture, designed to be clean, consistent and interpretability-friendly. 

For this demo notebook we'll look at GPT-2 Small, an 80M parameter model. To try the model the model out, let's find the loss on this paragraph!"""

loss = gpt2_small(model_description_text, return_type="loss")
print("Model loss:", loss)
# %%
print(gpt2_small.to_str_tokens("gpt2"))
print(gpt2_small.to_tokens("gpt2"))
print(gpt2_small.to_string([50256, 70, 457, 17]))
# %%
logits: Tensor = gpt2_small(model_description_text, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze()[:-1]
tokens = gpt2_small.to_tokens(model_description_text)
print(
    f"Correct guesses: {(prediction == tokens[:,1:]).sum()}, out of {len(tokens[0]) - 1}"
)
# %%
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)
attn_patterns_layer_0 = gpt2_cache["pattern", 0]
layer0_pattern_from_cache = gpt2_cache["pattern", 0]

# %%
seq, n_heads, d_head = gpt2_cache["q", 0].shape
layer0_pattern_from_q_and_k = einops.einsum(
    gpt2_cache["q", 0],
    gpt2_cache["k", 0],
    "seq_q n_heads d_head, seq_k n_heads d_head -> n_heads seq_q seq_k",
)

# scale
layer0_pattern_from_q_and_k = layer0_pattern_from_q_and_k / t.sqrt(
    t.tensor([gpt2_small.cfg.d_head])
)

# mask
mask = t.triu(t.ones((seq, seq)) == 1, diagonal=1)
layer0_pattern_from_q_and_k.masked_fill_(mask, -1e10)

# softmax
layer0_pattern_from_q_and_k = layer0_pattern_from_q_and_k.softmax(dim=-1)

t.testing.assert_close(layer0_pattern_from_cache, layer0_pattern_from_q_and_k)
print("Tests passed!")

# %%
print(type(gpt2_cache))
attention_pattern = gpt2_cache["pattern", 0]
print(attention_pattern.shape)
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=gpt2_str_tokens,
        attention=attention_pattern,
        # attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)
# %%
cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True,  # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer",
)
# %%
weights_dir = (section_dir / "attn_only_2L_half.pth").resolve()

if not weights_dir.exists():
    url = "https://drive.google.com/uc?id=1vcZLJnJoYKQs-2KOjkd6LvHZrkSdoxhu"
    output = str(weights_dir)
    gdown.download(url, output)
# %%
model = HookedTransformer(cfg)
pretrained_weights = t.load(weights_dir, map_location=device)
model.load_state_dict(pretrained_weights)

# %%
text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)
# %%
attention_pattern = cache["pattern", 1]
print(attention_pattern.shape)
model_str_tokens = model.to_str_tokens(text)

print("Layer 0 Head Attention Patterns:")
display(
    cv.attention.attention_patterns(
        tokens=model_str_tokens,
        attention=attention_pattern,
        attention_head_names=[f"L0H{i}" for i in range(12)],
    )
)


# %%
def current_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be current-token heads
    """
    detect_list = []
    for layer in range(len(model.blocks)):
        for head_i, head in enumerate(cache["pattern", layer]):
            count = 0
            seq = len(head)

            for i, attention in enumerate(head):
                if attention.argmax().item() == i:
                    count += 1
            if (count / seq) > 0.4:
                detect_list.append(f"{layer}.{head_i}")
    return detect_list


def prev_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    """
    detect_list = []
    for layer in range(len(model.blocks)):
        for head_i, head in enumerate(cache["pattern", layer]):
            count = 0
            seq = len(head)

            for i, attention in enumerate(head):
                if attention.argmax().item() == (i - 1):
                    count += 1
            if (count / seq) > 0.4:
                detect_list.append(f"{layer}.{head_i}")
    return detect_list


def first_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be first-token heads
    """
    detect_list = []
    for layer in range(len(model.blocks)):
        for head_i, head in enumerate(cache["pattern", layer]):
            count = 0
            seq = len(head)

            for i, attention in enumerate(head):
                if attention.argmax().item() == 0:
                    count += 1
            if (count / seq) > 0.8:
                detect_list.append(f"{layer}.{head_i}")
    return detect_list


print("Heads attending to current token  = ", ", ".join(current_attn_detector(cache)))
print("Heads attending to previous token = ", ", ".join(prev_attn_detector(cache)))
print("Heads attending to first token    = ", ", ".join(first_attn_detector(cache)))


# %%
def generate_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Int[Tensor, "batch full_seq_len"]:
    """
    Generates a sequence of repeated random tokens

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
    """
    prefix = (t.ones(batch, 1) * model.tokenizer.bos_token_id).long()
    # SOLUTION
    rep_tokens_half = t.randint(0, model.cfg.d_vocab, (batch, seq_len), dtype=t.int64)
    rep_tokens = t.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1).to(device)
    return rep_tokens


def run_and_cache_model_repeated_tokens(
    model: HookedTransformer, seq_len: int, batch: int = 1
) -> Tuple[t.Tensor, t.Tensor, ActivationCache]:
    """
    Generates a sequence of repeated random tokens, and runs the model on it, returning logits, tokens and cache

    Should use the `generate_repeated_tokens` function above

    Outputs are:
        rep_tokens: [batch, 1+2*seq_len]
        rep_logits: [batch, 1+2*seq_len, d_vocab]
        rep_cache: The cache of the model run on rep_tokens
    """
    # SOLUTION
    rep_tokens = generate_repeated_tokens(model, seq_len, batch)
    rep_logits, rep_cache = model.run_with_cache(rep_tokens)
    return rep_tokens, rep_logits, rep_cache


seq_len = 50
batch = 1
(rep_tokens, rep_logits, rep_cache) = run_and_cache_model_repeated_tokens(
    model, seq_len, batch
)
rep_cache.remove_batch_dim()
rep_str = model.to_str_tokens(rep_tokens)
model.reset_hooks()
log_probs = get_log_probs(rep_logits, rep_tokens).squeeze()

print(f"Performance on the first half: {log_probs[:seq_len].mean():.3f}")
print(f"Performance on the second half: {log_probs[seq_len:].mean():.3f}")

plot_loss_difference(log_probs, rep_str, seq_len)


# %%
def induction_attn_detector(cache: ActivationCache) -> List[str]:
    """
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be induction heads

    Remember - the tokens used to generate rep_cache are (bos_token, *rand_tokens, *rand_tokens)
    """
    # SOLUTION
    attn_heads = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][head]
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1] - 1) // 2
            score = attention_pattern.diagonal(-seq_len + 1).mean()
            if score > 0.4:
                attn_heads.append(f"{layer}.{head}")
    return attn_heads


print("Induction heads = ", ", ".join(induction_attn_detector(rep_cache)))


# %%
def hook_function(
    attn_pattern: Float[Tensor, "batch heads seqQ seqK"], hook: HookPoint
) -> Float[Tensor, "batch heads seqQ seqK"]:
    # modify attn_pattern (can be inplace)
    return attn_pattern


loss = model.run_with_hooks(
    tokens,
    return_type="loss",
    fwd_hooks=[("blocks.1.attn.hook_pattern", hook_function)],
)
# %%
seq_len = 50
batch = 10
rep_tokens_10 = generate_repeated_tokens(model, seq_len, batch)

# We make a tensor to store the induction score for each head.
# We put it on the model's device to avoid needing to move things between the GPU and CPU, which can be slow.
induction_score_store = t.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
)


def induction_score_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    """
    Calculates the induction score, and stores it in the [layer, head] position of the `induction_score_store` tensor.
    """
    # SOLUTION
    # Take the diagonal of attn paid from each dest posn to src posns (seq_len-1) tokens back
    # (This only has entries for tokens with index>=seq_len)
    induction_stripe = pattern.diagonal(dim1=-2, dim2=-1, offset=1 - seq_len)
    # Get an average score per head
    induction_score = einops.reduce(
        induction_stripe, "batch head_index position -> head_index", "mean"
    )
    # Store the result.
    induction_score_store[hook.layer(), :] = induction_score


# We make a boolean filter on activation names, that's true only on attention pattern names


pattern_hook_names_filter = lambda name: name.endswith("pattern")

# Run with hooks (this is where we write to the `induction_score_store` tensor`)
model.run_with_hooks(
    rep_tokens_10,
    return_type=None,  # For efficiency, we don't need to calculate the logits
    fwd_hooks=[(pattern_hook_names_filter, induction_score_hook)],
)

# Plot the induction scores for each head in each layer
imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=400,
)


# %%
def visualize_pattern_hook(
    pattern: Float[Tensor, "batch head_index dest_pos source_pos"],
    hook: HookPoint,
):
    print("Layer: ", hook.layer())
    display(
        cv.attention.attention_patterns(
            tokens=gpt2_small.to_str_tokens(rep_tokens[0]), attention=pattern.mean(0)
        )
    )


# YOUR CODE HERE - find induction heads in gpt2_small
# %%

induction_score_store = t.zeros(
    (gpt2_small.cfg.n_layers, gpt2_small.cfg.n_heads), device=gpt2_small.cfg.device
)

for induction_head_layer in [5, 6, 7]:
    gpt2_small.run_with_hooks(
        rep_tokens,
        return_type=None,  # For efficiency, we don't need to calculate the logits
        fwd_hooks=[
            (pattern_hook_names_filter, induction_score_hook),
            (
                utils.get_act_name("pattern", induction_head_layer),
                visualize_pattern_hook,
            ),
        ],
    )

imshow(
    induction_score_store,
    labels={"x": "Head", "y": "Layer"},
    title="Induction Score by Head",
    text_auto=".2f",
    width=900,
    height=400,
)


# %%
def logit_attribution(
    embed: Float[Tensor, "seq d_model"],
    l1_results: Float[Tensor, "seq nheads d_model"],
    l2_results: Float[Tensor, "seq nheads d_model"],
    W_U: Float[Tensor, "d_model d_vocab"],
    tokens: Int[Tensor, "seq"],
) -> Float[Tensor, "seq-1 n_components"]:
    """
    Inputs:
        embed: the embeddings of the tokens (i.e. token + position embeddings)
        l1_results: the outputs of the attention heads at layer 1 (with head as one of the dimensions)
        l2_results: the outputs of the attention heads at layer 2 (with head as one of the dimensions)
        W_U: the unembedding matrix
        tokens: the token ids of the sequence

    Returns:
        Tensor of shape (seq_len-1, n_components)
        represents the concatenation (along dim=-1) of logit attributions from:
            the direct path (seq-1,1)
            layer 0 logits (seq-1, n_heads)
            layer 1 logits (seq-1, n_heads)
        so n_components = 1 + 2*n_heads
    """
    W_U_correct_tokens = W_U[:, tokens[1:]]
    embed_contribution = einops.einsum(
        embed[:-1], W_U_correct_tokens, "seq emb, emb seq -> seq "
    ).unsqueeze(-1)
    l1_contribution = einops.einsum(
        l1_results[:-1], W_U_correct_tokens, "seq nheads emb, emb seq -> seq nheads"
    )
    l2_contribution = einops.einsum(
        l2_results[:-1], W_U_correct_tokens, "seq nheads emb, emb seq -> seq nheads"
    )
    return t.concat((embed_contribution, l1_contribution, l2_contribution), dim=-1)


text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."
logits, cache = model.run_with_cache(text, remove_batch_dim=True)
str_tokens = model.to_str_tokens(text)
tokens = model.to_tokens(text)

with t.inference_mode():
    embed = cache["embed"]
    l1_results = cache["result", 0]
    l2_results = cache["result", 1]
    logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])
    # Uses fancy indexing to get a len(tokens[0])-1 length tensor, where the kth entry is the predicted logit for the correct k+1th token
    correct_token_logits = logits[0, t.arange(len(tokens[0]) - 1), tokens[0, 1:]]
    t.testing.assert_close(logit_attr.sum(1), correct_token_logits, atol=1e-3, rtol=0)
    print("Tests passed!")

# %%
embed = cache["embed"]
l1_results = cache["result", 0]
l2_results = cache["result", 1]
logit_attr = logit_attribution(embed, l1_results, l2_results, model.W_U, tokens[0])

plot_logit_attribution(model, logit_attr, tokens)


# %%
def head_ablation_hook(
    v: Float[Tensor, "batch seq n_heads d_head"],
    hook: HookPoint,
    head_index_to_ablate: int,
) -> Float[Tensor, "batch seq n_heads d_head"]:
    # return t.zeros_like(v[:,:,head_index_to_ablate,:].unsqueeze(-2))
    v[:, :, head_index_to_ablate, :] = t.zeros_like(v[:, :, head_index_to_ablate, :])


def cross_entropy_loss(logits, tokens):
    """
    Computes the mean cross entropy between logits (the model's prediction) and tokens (the true values).

    (optional, you can just use return_type="loss" instead.)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    pred_log_probs = t.gather(log_probs[:, :-1], -1, tokens[:, 1:, None])[..., 0]
    return -pred_log_probs.mean()


def get_ablation_scores(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"]
) -> Float[Tensor, "n_layers n_heads"]:
    """
    Returns a tensor of shape (n_layers, n_heads) containing the increase in cross entropy loss from ablating the output of each head.
    """
    # Initialize an object to store the ablation scores
    ablation_scores = t.zeros(
        (model.cfg.n_layers, model.cfg.n_heads), device=model.cfg.device
    )

    # Calculating loss without any ablation, to act as a baseline
    model.reset_hooks()
    logits = model(tokens, return_type="logits")
    loss_no_ablation = cross_entropy_loss(logits, tokens)

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            # Use functools.partial to create a temporary hook function with the head number fixed
            temp_hook_fn = functools.partial(
                head_ablation_hook, head_index_to_ablate=head
            )
            # Run the model with the ablation hook
            ablated_logits = model.run_with_hooks(
                tokens, fwd_hooks=[(utils.get_act_name("v", layer), temp_hook_fn)]
            )
            # Calculate the logit difference
            loss = cross_entropy_loss(ablated_logits, tokens)
            # Store the result, subtracting the clean loss so that a value of zero means no change in loss
            ablation_scores[layer, head] = loss - loss_no_ablation

    return ablation_scores


ablation_scores = get_ablation_scores(model, rep_tokens)
tests.test_get_ablation_scores(ablation_scores, model, rep_tokens)
# %%
imshow(
    ablation_scores,
    labels={"x": "Head", "y": "Layer", "color": "Logit diff"},
    title="Logit Difference After Ablating Heads",
    text_auto=".2f",
    width=900,
    height=400,
)
# %%
A = t.randn(5, 2)
B = t.randn(2, 5)
AB = A @ B
AB_factor = FactoredMatrix(A, B)
print("Norms:")
print(AB.norm())
print(AB_factor.norm())

print(
    f"Right dimension: {AB_factor.rdim}, Left dimension: {AB_factor.ldim}, Hidden dimension: {AB_factor.mdim}"
)
# %%
print("Eigenvalues:")
print(t.linalg.eig(AB).eigenvalues)
print(AB_factor.eigenvalues)
print()
print("Singular Values:")
print(t.linalg.svd(AB).S)
print(AB_factor.S)
print("Full SVD:")
print(AB_factor.svd())
# %%
C = t.randn(5, 300)
ABC = AB @ C
ABC_factor = AB_factor @ C
print("Unfactored:", ABC.shape, ABC.norm())
print("Factored:", ABC_factor.shape, ABC_factor.norm())
print(
    f"Right dimension: {ABC_factor.rdim}, Left dimension: {ABC_factor.ldim}, Hidden dimension: {ABC_factor.mdim}"
)
# %%
layer, head_index = 1, 4
full_OV_circuit = (
    FactoredMatrix(model.W_E, model.W_V[1, 4]) @ model.W_O[1, 4] @ model.W_U
)
tests.test_full_OV_circuit(full_OV_circuit, model, layer, head_index)
# %%

imshow(
    full_OV_circuit_sample,
    labels={"x": "Input token", "y": "Logits on output token"},
    title="Full OV circuit for copying head",
    width=700,
)


# %%
def mask_scores(attn_scores: Float[Tensor, "query_nctx key_nctx"]):
    """Mask the attention scores so that tokens don't attend to previous tokens."""
    assert attn_scores.shape == (model.cfg.n_ctx, model.cfg.n_ctx)
    mask = t.tril(t.ones_like(attn_scores)).bool()
    neg_inf = t.tensor(-1.0e6).to(attn_scores.device)
    masked_attn_scores = t.where(mask, attn_scores, neg_inf)
    return masked_attn_scores


layer = 0
head_index = 7

W_K = model.W_K[layer, head_index]
W_Q = model.W_Q[layer, head_index]

attention_scores = W_Q @ W_K.T
attention_scores = attention_scores / model.cfg.d_head**0.5

pos_by_pos_pattern = model.W_pos @ attention_scores @ model.W_pos.T
# YOUR CODE HERE - calculate the matrix `pos_by_pos_pattern` as described above
pos_by_pos_pattern = mask_scores(pos_by_pos_pattern).softmax(dim=-1)
tests.test_pos_by_pos_pattern(pos_by_pos_pattern, model, layer, head_index)

# %%
print(f"Avg lower-diagonal value: {pos_by_pos_pattern.diag(-1).mean():.4f}")

imshow(
    utils.to_numpy(pos_by_pos_pattern[:100, :100]),
    labels={"x": "Key", "y": "Query"},
    title="Attention patterns for prev-token QK circuit, first 100 indices",
    width=700,
)


# %%
def decompose_qk_input(cache: ActivationCache) -> t.Tensor:
    """
    Output is decomposed_qk_input, with shape [2+num_heads, seq, d_model]

    The [i, :, :]th element is y_i (from notation above)
    """
    # SOLUTION
    y0 = cache["embed"].unsqueeze(0)  # shape (1, seq, d_model)
    y1 = cache["pos_embed"].unsqueeze(0)  # shape (1, seq, d_model)
    y_rest = cache["result", 0].transpose(0, 1)  # shape (12, seq, d_model)

    return t.concat([y0, y1, y_rest], dim=0)


def decompose_q(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    """
    Output is decomposed_q with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_Q (so the sum along axis 0 is just the q-values)
    """
    # SOLUTION
    W_Q = model.W_Q[1, ind_head_index]

    return einops.einsum(
        decomposed_qk_input, W_Q, "n seq d_head, d_head d_model -> n seq d_model"
    )


def decompose_k(decomposed_qk_input: t.Tensor, ind_head_index: int) -> t.Tensor:
    """
    Output is decomposed_k with shape [2+num_heads, position, d_head]

    The [i, :, :]th element is y_i @ W_K(so the sum along axis 0 is just the k-values)
    """
    # SOLUTION
    W_K = model.W_K[1, ind_head_index]

    return einops.einsum(
        decomposed_qk_input, W_K, "n seq d_head, d_head d_model -> n seq d_model"
    )


ind_head_index = 4
# First we get decomposed q and k input, and check they're what we expect
decomposed_qk_input = decompose_qk_input(rep_cache)
decomposed_q = decompose_q(decomposed_qk_input, ind_head_index)
decomposed_k = decompose_k(decomposed_qk_input, ind_head_index)
t.testing.assert_close(
    decomposed_qk_input.sum(0),
    rep_cache["resid_pre", 1] + rep_cache["pos_embed"],
    rtol=0.01,
    atol=1e-05,
)
t.testing.assert_close(
    decomposed_q.sum(0), rep_cache["q", 1][:, ind_head_index], rtol=0.01, atol=0.001
)
t.testing.assert_close(
    decomposed_k.sum(0), rep_cache["k", 1][:, ind_head_index], rtol=0.01, atol=0.01
)
# Second, we plot our results

component_labels = ["Embed", "PosEmbed"] + [f"0.{h}" for h in range(model.cfg.n_heads)]
for decomposed_input, name in [(decomposed_q, "query"), (decomposed_k, "key")]:
    imshow(
        utils.to_numpy(decomposed_input.pow(2).sum([-1])),
        labels={"x": "Position", "y": "Component"},
        title=f"Norms of components of {name}",
        y=component_labels,
        width=1000,
        height=400,
    )


# %%
def decompose_attn_scores(decomposed_q: t.Tensor, decomposed_k: t.Tensor) -> t.Tensor:
    """
    Output is decomposed_scores with shape [query_component, key_component, query_pos, key_pos]

    The [i, j, :, :]th element is y_i @ W_QK @ y_j^T (so the sum along both first axes are the attention scores)
    """
    output = einops.einsum(
        decomposed_q,
        decomposed_k,
        "component_q seq_q dhead, component_k seq_k dhead -> component_q component_k seq_q seq_k",
    )
    return output


tests.test_decompose_attn_scores(decompose_attn_scores, decomposed_q, decomposed_k)

# %%
decomposed_scores = decompose_attn_scores(decomposed_q, decomposed_k)
decomposed_stds = einops.reduce(
    decomposed_scores,
    "query_decomp key_decomp query_pos key_pos -> query_decomp key_decomp",
    t.std,
)

# First plot: attention score contribution from (query_component, key_component) = (Embed, L0H7)
imshow(
    utils.to_numpy(t.tril(decomposed_scores[0, 9])),
    title="Attention score contributions from (query, key) = (embed, output of L0H7)",
    width=800,
)

# Second plot: std dev over query and key positions, shown by component
imshow(
    utils.to_numpy(decomposed_stds),
    labels={"x": "Key Component", "y": "Query Component"},
    title="Standard deviations of attention score contributions (by key and query component)",
    x=component_labels,
    y=component_labels,
    width=800,
)


# %%
def find_K_comp_full_circuit(
    model: HookedTransformer, prev_token_head_index: int, ind_head_index: int
) -> FactoredMatrix:
    """
    Returns a (vocab, vocab)-size FactoredMatrix, with the first dimension being the query side and the second dimension being the key side (going via the previous token head)
    """
    full_circuit = (
        FactoredMatrix(model.W_E, model.W_Q[1, ind_head_index])
        @ model.W_K[1, ind_head_index].T
        @ (model.W_V[0, prev_token_head_index] @ model.W_O[0, prev_token_head_index]).T
        @ model.W_E.T
    )
    return full_circuit


prev_token_head_index = 7
ind_head_index = 10
K_comp_circuit = find_K_comp_full_circuit(model, prev_token_head_index, ind_head_index)


def top_1_acc(full_OV_circuit: FactoredMatrix) -> float:
    """
    This should take the argmax of each column (ie over dim=0) and return the fraction of the time that's equal to the correct logit
    """
    # SOLUTION
    AB = full_OV_circuit.AB

    return (
        (t.argmax(AB, dim=1) == t.arange(AB.shape[0]).to(device)).float().mean().item()
    )


tests.test_find_K_comp_full_circuit(find_K_comp_full_circuit, model)

print(
    f"Fraction of tokens where the highest activating key is the same token: {top_1_acc(K_comp_circuit.T):.4f}"
)


# %%
# Get all QK and OV matrices
def get_comp_score(
    W_A: Float[Tensor, "in_A out_A"], W_B: Float[Tensor, "out_A out_B"]
) -> float:
    """
    Return the composition score between W_A and W_B.
    """
    # SOLUTION
    W_A_norm = W_A.pow(2).sum().sqrt()
    W_B_norm = W_B.pow(2).sum().sqrt()
    W_AB_norm = (W_A @ W_B).pow(2).sum().sqrt()

    return (W_AB_norm / (W_A_norm * W_B_norm)).item()


# Get all QK and OV matrices
W_QK = model.W_Q @ model.W_K.transpose(-1, -2)
W_OV = model.W_V @ model.W_O

# Define tensors to hold the composition scores
composition_scores = {
    "Q": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
    "K": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
    "V": t.zeros(model.cfg.n_heads, model.cfg.n_heads).to(device),
}

# Fill in the tensors, by looping over W_A and W_B from layers 0 and 1
for i in tqdm(range(model.cfg.n_heads)):
    for j in range(model.cfg.n_heads):
        composition_scores["Q"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j])
        composition_scores["K"][i, j] = get_comp_score(W_OV[0, i], W_QK[1, j].T)
        composition_scores["V"][i, j] = get_comp_score(W_OV[0, i], W_OV[1, j])
# %%
for comp_type in "QKV":
    plot_comp_scores(
        model, composition_scores[comp_type], f"{comp_type} Composition Scores"
    ).show()


# %%


def generate_single_random_comp_score() -> float:
    """
    Write a function which generates a single composition score for random matrices
    """
    # SOLUTION
    W_A_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_left = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_A_right = t.empty(model.cfg.d_model, model.cfg.d_head)
    W_B_right = t.empty(model.cfg.d_model, model.cfg.d_head)

    for W in [W_A_left, W_B_left, W_A_right, W_B_right]:
        nn.init.kaiming_uniform_(W, a=np.sqrt(5))

    W_A = W_A_left @ W_A_right.T
    W_B = W_B_left @ W_B_right.T

    return get_comp_score(W_A, W_B)


n_samples = 300
comp_scores_baseline = np.zeros(n_samples)
for i in tqdm(range(n_samples)):
    comp_scores_baseline[i] = generate_single_random_comp_score()
print("\nMean:", comp_scores_baseline.mean())
print("Std:", comp_scores_baseline.std())
hist(
    comp_scores_baseline,
    nbins=50,
    width=800,
    labels={"x": "Composition score"},
    title="Random composition scores",
)
# %%
baseline = comp_scores_baseline.mean()
for comp_type, comp_scores in composition_scores.items():
    plot_comp_scores(
        model, comp_scores, f"{comp_type} Composition Scores", baseline=baseline
    )


# %%
def ablation_induction_score(
    prev_head_index: Optional[int], ind_head_index: int
) -> float:
    """
    Takes as input the index of the L0 head and the index of the L1 head, and then runs with the previous token head ablated and returns the induction score for the ind_head_index now.
    """

    def ablation_hook(v, hook):
        if prev_head_index is not None:
            v[:, :, prev_head_index] = 0.0
        return v

    def induction_pattern_hook(attn, hook):
        hook.ctx[prev_head_index] = attn[0, ind_head_index].diag(-(seq_len - 1)).mean()

    model.run_with_hooks(
        rep_tokens,
        fwd_hooks=[
            (utils.get_act_name("v", 0), ablation_hook),
            (utils.get_act_name("pattern", 1), induction_pattern_hook),
        ],
    )
    return model.blocks[1].attn.hook_pattern.ctx[prev_head_index].item()


baseline_induction_score = ablation_induction_score(None, 4)
print(f"Induction score for no ablations: {baseline_induction_score:.5f}\n")
for i in range(model.cfg.n_heads):
    new_induction_score = ablation_induction_score(i, 4)
    induction_score_change = new_induction_score - baseline_induction_score
    print(f"Ablation score change for head {i:02}: {induction_score_change:+.5f}")

# %%
# RUN PCA on gpt2_cache, recreating the plot from https://www.lesswrong.com/posts/X26ksz4p3wSyycKNB/gears-level-mental-models-of-transformer-interpretability#Residual_Stream_as_Output_Accumulation:~:text=The%20Models-,Residual%20Stream%20as%20Output%20Accumulation,-The%20residual%20stream

from sklearn.decomposition import PCA
import pandas as pd

resid_post = [gpt2_cache["resid_post", i] for i in range(12)]
resids = t.stack(resid_post)
resids = einops.rearrange(resids, "layers context dmodel -> context layers dmodel")

results_list = []
for i in range(resids.shape[0]):
    pca = PCA(n_components=10)
    pca.fit(resids[i])
    results_list.append(pca.explained_variance_ratio_)

results = np.array(results_list)
mean = results.mean(axis=0)
std = results.std(axis=0)
df = pd.DataFrame([mean, std]).T
df.columns = ["y", "error_y"]

fig = px.bar(df, y="y", error_y="error_y")
fig.show()
# %%
