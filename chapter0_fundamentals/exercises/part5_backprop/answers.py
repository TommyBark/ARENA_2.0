# %%
import os
import sys
import re
import time
import torch as t
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Iterable, Optional, Union, Dict, List, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm

Arr = np.ndarray
grad_tracking_enabled = True

# Make sure exercises are in the path
chapter = r"chapter0_fundamentals"
exercises_dir = Path(f"{os.getcwd().split(chapter)[0]}/{chapter}/exercises").resolve()
section_dir = exercises_dir / "part5_backprop"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
os.chdir(section_dir)

import part5_backprop.tests as tests
from part5_backprop.utils import visualize, get_mnist
from plotly_utils import line

MAIN = __name__ == "__main__"


# %%
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    """
    return grad_out / x


if MAIN:
    tests.test_log_back(log_back)


# %%
def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    """
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.

    original -> broadcasted
    (4,) -> (3,4)  -> output 0
    (1,4) -> (3,4) -> output 0
    (1,8,9) -> (2,3,4,8,9)
    (1,1,8,9) -> (2,3,4,5,8,9)
    """
    o_shape = original.shape
    b_shape = broadcasted.shape
    l_b = len(b_shape)
    l_o = len(o_shape)
    # keepdim = True if len(o_shape) == len(b_shape) else False

    sum_dims = [*range(l_b - l_o)]
    for i in range(-1, -l_o - 1, -1):
        if o_shape[i] != b_shape[i]:
            sum_dims.append(i)

    unbroad = broadcasted.sum(axis=tuple(sum_dims))
    return unbroad.reshape(o_shape)


if MAIN:
    print(unbroadcast(np.zeros((2, 3, 4, 8, 9)), np.zeros((1, 8, 9))))
    tests.test_unbroadcast(unbroadcast)


# %%
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    """Backwards function for x * y wrt argument 0 aka x."""
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(grad_out * y, x)


def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    """Backwards function for x * y wrt argument 1 aka y."""
    if not isinstance(x, Arr):
        x = np.array(x)
    return unbroadcast(x * grad_out, y)


if MAIN:
    tests.test_multiply_back(multiply_back0, multiply_back1)
    tests.test_multiply_back_float(multiply_back0, multiply_back1)


# %%
def forward_and_back(a: Arr, b: Arr, c: Arr) -> Tuple[Arr, Arr, Arr]:
    """
    Calculates the output of the computational graph above (g), then backpropogates the gradients and returns dg/da, dg/db, and dg/dc
    """
    d = a * b
    e = np.log(c)
    f = d * e
    g = np.log(f)

    dg_dg = np.array([1])
    dg_df = log_back(dg_dg, 1, f)
    dg_de = multiply_back1(dg_df, f, d, e)
    dg_dd = multiply_back0(dg_df, f, d, e)
    dg_dc = log_back(dg_de, 1, c)
    dg_da = multiply_back0(dg_dd, d, a, b)
    dg_db = multiply_back1(dg_dd, d, a, b)
    return dg_da, dg_db, dg_dc


if MAIN:
    tests.test_forward_and_back(forward_and_back)


# %%
@dataclass(frozen=True)
class Recipe:
    """Extra information necessary to run backpropagation. You don't need to modify this."""

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll create for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: Dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: Dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."


# %%
class BackwardFuncLookup:
    def __init__(self) -> None:
        self.func_dict = {}

    def add_back_func(
        self, forward_fn: Callable, arg_position: int, back_fn: Callable
    ) -> None:
        self.func_dict[(forward_fn, arg_position)] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        return self.func_dict[(forward_fn, arg_position)]


if MAIN:
    BACK_FUNCS = BackwardFuncLookup()
    BACK_FUNCS.add_back_func(np.log, 0, log_back)
    BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
    BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

    assert BACK_FUNCS.get_back_func(np.log, 0) == log_back
    assert BACK_FUNCS.get_back_func(np.multiply, 0) == multiply_back0
    assert BACK_FUNCS.get_back_func(np.multiply, 1) == multiply_back1

    print("Tests passed - BackwardFuncLookup class is working as expected!")

# %%
Arr = np.ndarray


class Tensor:
    """
    A drop-in replacement for torch.Tensor supporting a subset of features.
    """

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other) -> "Tensor":
        return multiply(other, self)

    def __truediv__(self, other) -> "Tensor":
        return true_divide(self, other)

    def __rtruediv__(self, other) -> "Tensor":
        return true_divide(other, self)

    def __matmul__(self, other) -> "Tensor":
        return matmul(self, other)

    def __rmatmul__(self, other) -> "Tensor":
        return matmul(other, self)

    def __eq__(self, other) -> "Tensor":
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
        add_(self, other, alpha=alpha)
        return self

    @property
    def T(self) -> "Tensor":
        return permute(self)

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self):
        return log(self)

    def exp(self):
        return exp(self)

    def reshape(self, new_shape):
        return reshape(self, new_shape)

    def expand(self, new_shape):
        return expand(self, new_shape)

    def permute(self, dims):
        return permute(self, dims)

    def maximum(self, other):
        return maximum(self, other)

    def relu(self):
        return relu(self)

    def argmax(self, dim=None, keepdim=False):
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        """Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html"""
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError(
                "bool value of Tensor with more than one value is ambiguous"
            )
        return bool(self.item())


def empty(*shape: int) -> Tensor:
    """Like torch.empty."""
    return Tensor(np.empty(shape))


def zeros(*shape: int) -> Tensor:
    """Like torch.zeros."""
    return Tensor(np.zeros(shape))


def arange(start: int, end: int, step=1) -> Tensor:
    """Like torch.arange(start, end)."""
    return Tensor(np.arange(start, end, step=step))


def tensor(array: Arr, requires_grad=False) -> Tensor:
    """Like torch.tensor."""
    return Tensor(array, requires_grad=requires_grad)


# %%
def log_forward(x: Tensor) -> Tensor:
    """Performs np.log on a Tensor object."""

    log = np.log(x.array)
    requires_grad = grad_tracking_enabled and (
        x.requires_grad or (x.recipe is not None)
    )
    log_tensor = Tensor(log, requires_grad=requires_grad)
    if requires_grad:
        log_tensor.recipe = Recipe(np.log, (x.array,), {}, {0: x})
    return log_tensor


if MAIN:
    log = log_forward
    tests.test_log(Tensor, log_forward)
    tests.test_log_no_grad(Tensor, log_forward)
    a = Tensor([1], requires_grad=True)
    grad_tracking_enabled = False
    b = log_forward(a)
    grad_tracking_enabled = True
    assert (
        not b.requires_grad
    ), "should not require grad if grad tracking globally disabled"
    assert (
        b.recipe is None
    ), "should not create recipe if grad tracking globally disabled"


# %%
def multiply_forward(a: Union[Tensor, int], b: Union[Tensor, int]) -> Tensor:
    """Performs np.multiply on a Tensor object."""
    a_ten = isinstance(a, Tensor)
    b_ten = isinstance(b, Tensor)
    assert a_ten or b_ten

    a_val = a.array if a_ten else a
    b_val = b.array if b_ten else b

    mult = np.multiply(a_val, b_val)

    requires_grad = False
    if a_ten:
        requires_grad = requires_grad or a.requires_grad
    if b_ten:
        requires_grad = requires_grad or b.requires_grad

    requires_grad = grad_tracking_enabled and requires_grad
    mult_tensor = Tensor(mult, requires_grad=requires_grad)
    if requires_grad:
        parents = {
            idx: arr for idx, arr in enumerate([a, b]) if isinstance(arr, Tensor)
        }
        mult_tensor.recipe = Recipe(np.multiply, (a_val, b_val), {}, parents)
    return mult_tensor


if MAIN:
    multiply = multiply_forward
    tests.test_multiply(Tensor, multiply_forward)
    tests.test_multiply_no_grad(Tensor, multiply_forward)
    tests.test_multiply_float(Tensor, multiply_forward)
    a = Tensor([2], requires_grad=True)
    b = Tensor([3], requires_grad=True)
    grad_tracking_enabled = False
    b = multiply_forward(a, b)
    grad_tracking_enabled = True
    assert (
        not b.requires_grad
    ), "should not require grad if grad tracking globally disabled"
    assert (
        b.recipe is None
    ), "should not create recipe if grad tracking globally disabled"


# %%
def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    """
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and
        any number of keyword arguments which we aren't allowing to be NumPy arrays at
        present. It returns a single NumPy array.

    is_differentiable:
        if True, numpy_func is differentiable with respect to some input argument, so we
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array,
        this has a Tensor instead.
    """

    def tensor_func(*args: Any, **kwargs: Any) -> Tensor:
        another_args = [arg.array if isinstance(arg, Tensor) else arg for arg in args]
        out = numpy_func(*another_args, **kwargs)

        requires_grad_args = np.array(
            [
                tens.requires_grad or tens.recipe is not None
                for tens in args
                if isinstance(tens, Tensor)
            ]
        ).any()
        requires_grad = grad_tracking_enabled and requires_grad_args

        out = Tensor(out, requires_grad)
        if is_differentiable and requires_grad:
            out.recipe = Recipe(
                numpy_func,
                tuple(another_args),
                kwargs,
                {i: arg for i, arg in enumerate(args) if isinstance(arg, Tensor)},
            )
        return out

    return tensor_func


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    # need to be careful with sum, because kwargs have different names in torch and numpy
    return np.sum(x, axis=dim, keepdims=keepdim)


if MAIN:
    log = wrap_forward_fn(np.log)
    multiply = wrap_forward_fn(np.multiply)
    eq = wrap_forward_fn(np.equal, is_differentiable=False)
    sum = wrap_forward_fn(_sum)

    tests.test_log(Tensor, log)
    tests.test_log_no_grad(Tensor, log)
    tests.test_multiply(Tensor, multiply)
    tests.test_multiply_no_grad(Tensor, multiply)
    tests.test_multiply_float(Tensor, multiply)
    tests.test_sum(Tensor)


# %%
class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> List[Node]:
    return node.children


def topological_sort(node: Node, get_children: Callable) -> List[Node]:
    """
    Return a list of node's descendants in reverse topological order from future to past (i.e. `node` should be last).

    Should raise an error if the graph with `node` as root is not in fact acyclic.
    """
    # SOLUTION

    result: List[
        Node
    ] = []  # stores the list of nodes to be returned (in reverse topological order)
    perm: set[
        Node
    ] = set()  # same as `result`, but as a set (faster to check for membership)
    temp: set[
        Node
    ] = set()  # keeps track of previously visited nodes (to detect cyclicity)

    def visit(cur: Node):
        """
        Recursive function which visits all the children of the current node, and appends them all
        to `result` in the order they were found.
        """
        if cur in perm:
            return
        if cur in temp:
            raise ValueError("Not a DAG!")
        temp.add(cur)

        for next in get_children(cur):
            visit(next)

        result.append(cur)
        perm.add(cur)
        temp.remove(cur)

    visit(node)
    return result


# %%
def sorted_computational_graph(tensor: Tensor) -> List[Tensor]:
    """
    For a given tensor, return a list of Tensors that make up the nodes of the given Tensor's computational graph,
    in reverse topological order (i.e. `tensor` should be first).
    """

    # SOLUTION
    def get_parents(tensor: Tensor) -> List[Tensor]:
        if tensor.recipe is None:
            return []
        return list(tensor.recipe.parents.values())

    return topological_sort(tensor, get_parents)[::-1]


if MAIN:
    a = Tensor([1], requires_grad=True)
    b = Tensor([2], requires_grad=True)
    c = Tensor([3], requires_grad=True)
    d = a * b
    e = c.log()
    f = d * e
    g = f.log()
    name_lookup = {a: "a", b: "b", c: "c", d: "d", e: "e", f: "f", g: "g"}

    print([name_lookup[t] for t in sorted_computational_graph(g)])


# %%


def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    """Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node:
        The rightmost node in the computation graph.
        If it contains more than one element, end_grad must be provided.
    end_grad:
        A tensor of the same shape as end_node.
        Set to 1 if not specified and end_node has only one element.
    """
    assert len(end_node.shape) == 1 or end_grad is not None
    if end_grad is not None:
        assert end_node.shape == end_grad.shape

    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array
    grads: Dict[Tensor, Arr] = {end_node: end_grad_arr}

    # L = (end_grad_arr * end_node).sum()
    compute_graph = sorted_computational_graph(end_node)

    for tensor in compute_graph:
        outgrad = grads[tensor]
        if tensor.is_leaf and tensor.requires_grad:
            if tensor.grad is None:
                tensor.grad = Tensor(outgrad, requires_grad=False)
            else:
                tensor.grad.array += outgrad
        if tensor.recipe is None or tensor.recipe.parents is None:
            continue

        for i, parent in tensor.recipe.parents.items():
            b_f = BACK_FUNCS.get_back_func(tensor.recipe.func, i)
            in_grad = b_f(
                outgrad, tensor.array, *tensor.recipe.args, **tensor.recipe.kwargs
            )
            if parent in grads:
                grads[parent] += in_grad
            else:
                grads[parent] = in_grad


if MAIN:
    tests.test_backprop(Tensor)
    tests.test_backprop_branching(Tensor)
    tests.test_backprop_requires_grad_false(Tensor)
    tests.test_backprop_float_arg(Tensor)


# %%
def _argmax(x: Arr, dim=None, keepdim=False):
    """Like torch.argmax."""
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))


if MAIN:
    argmax = wrap_forward_fn(_argmax, is_differentiable=False)

    a = Tensor([1.0, 0.0, 3.0, 4.0], requires_grad=True)
    b = a.argmax()
    assert not b.requires_grad
    assert b.recipe is None
    assert b.item() == 3


# %%
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backward function for f(x) = -x elementwise."""
    return unbroadcast(-grad_out, x)


if MAIN:
    negative = wrap_forward_fn(np.negative)
    BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

    tests.test_negative_back(Tensor)


# %%
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return unbroadcast(grad_out * out, x)


if MAIN:
    exp = wrap_forward_fn(np.exp)
    BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

    tests.test_exp_back(Tensor)


def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    # SOLUTION
    return np.reshape(grad_out, x.shape)


# %%
def invert_transposition(axes: tuple) -> tuple:
    """
    axes: tuple indicating a transition

    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x

    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 1, 2)
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    """
    # SOLUTION

    # Slick solution:
    return tuple(np.argsort(axes))


def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, invert_transposition(axes))


if MAIN:
    BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
    permute = wrap_forward_fn(np.transpose)

    tests.test_permute_back(Tensor)

if MAIN:
    BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
    permute = wrap_forward_fn(np.transpose)

    tests.test_permute_back(Tensor)

# %%
if MAIN:
    x = np.array([1, 2, 3])

    np.broadcast_to(x, (3, 3))


# %%
def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)


def _expand(x: Arr, new_shape) -> Arr:
    """
    Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    """
    x_shape = x.shape
    expand_shape = list(new_shape)
    for i, j in enumerate(new_shape):
        if new_shape[-(i + 1)] == -1:
            expand_shape[-(i + 1)] = x_shape[-(i + 1)]

    return np.broadcast_to(x, expand_shape)


if MAIN:
    expand = wrap_forward_fn(_expand)
    BACK_FUNCS.add_back_func(_expand, 0, expand_back)

    tests.test_expand(Tensor)
    tests.test_expand_negative_length(Tensor)


# %%
def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    """Basic idea: repeat grad_out over the dims along which x was summed"""
    # SOLUTION

    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    if not isinstance(grad_out, Arr):
        grad_out = np.array(grad_out)

    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))

    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if keepdim == False:
        grad_out = np.expand_dims(grad_out, dim)

    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)


def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    """Like torch.sum, calling np.sum internally."""
    return np.sum(x, axis=dim, keepdims=keepdim)


if MAIN:
    sum = wrap_forward_fn(_sum)
    BACK_FUNCS.add_back_func(_sum, 0, sum_back)

    tests.test_sum_keepdim_false(Tensor)
    tests.test_sum_keepdim_true(Tensor)
    tests.test_sum_dim_none(Tensor)

# %%
BACK_FUNCS.add_back_func(
    np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y)
)
BACK_FUNCS.add_back_func(
    np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x)
)
BACK_FUNCS.add_back_func(
    np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y)
)
BACK_FUNCS.add_back_func(
    np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out / y, x)
)
BACK_FUNCS.add_back_func(
    np.true_divide,
    1,
    lambda grad_out, out, x, y: unbroadcast(grad_out * (-x / y**2), y),
)


# %%
class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        """Share the array with the provided tensor."""
        return super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"


# %%
class Module:
    _modules: Dict[str, "Module"]
    _parameters: Dict[str, Parameter]

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        """Return the direct child modules of this module."""
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """
        Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        """
        l = [j for k, j in self._parameters.items()]
        if recurse:
            for module in self.modules():
                l.extend(module.parameters(recurse=True))
        else:
            return iter(l)

    def __setattr__(self, key: str, val: Any) -> None:
        """
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        """
        if isinstance(val, Module):
            self._modules[key] = val
        elif isinstance(val, Parameter):
            self._parameters[key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union[Parameter, "Module"]:
        """
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        """
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        raise KeyError(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)

        lines = [
            f"({key}): {_indent(repr(module), 2)}"
            for key, module in self._modules.items()
        ]
        return "".join(
            [
                self.__class__.__name__ + "(",
                "\n  " + "\n  ".join(lines) + "\n" if lines else "",
                ")",
            ]
        )


class TestInnerModule(Module):
    def __init__(self):
        super().__init__()
        self.param1 = Parameter(Tensor([1.0]))
        self.param2 = Parameter(Tensor([2.0]))


class TestModule(Module):
    def __init__(self):
        super().__init__()
        self.inner = TestInnerModule()
        self.param3 = Parameter(Tensor([3.0]))


if MAIN:
    mod = TestModule()
    assert list(mod.modules()) == [mod.inner]
    assert list(mod.parameters()) == [
        mod.param3,
        mod.inner.param1,
        mod.inner.param2,
    ], "parameters should come before submodule parameters"
    print("Manually verify that the repr looks reasonable:")
    print(mod)


# %%
def _matmul2d(x: Arr, y: Arr) -> Arr:
    """Matrix multiply restricted to the case where both inputs are exactly 2D."""
    return x @ y


def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    # SOLUTION
    return grad_out @ y.T


def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    # SOLUTION
    return x.T @ grad_out


if MAIN:
    matmul = wrap_forward_fn(_matmul2d)
    BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
    BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

    tests.test_matmul2d(Tensor)

# %%
if MAIN:
    add = wrap_forward_fn(np.add)
    subtract = wrap_forward_fn(np.subtract)
    true_divide = wrap_forward_fn(np.true_divide)


    # Your code here - add to the BACK_FUNCS object
# %%
class Linear(Module):
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        # SOLUTION
        self.in_features = in_features
        self.out_features = out_features

        # sf needs to be a float
        sf = in_features**-0.5

        weight = sf * Tensor(2 * np.random.rand(out_features, in_features) - 1)
        self.weight = Parameter(weight)

        if bias:
            bias = sf * Tensor(
                2
                * np.random.rand(
                    out_features,
                )
                - 1
            )
            self.bias = Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        # SOLUTION
        out = x @ self.weight.T
        # Note, transpose has been defined as .permute(-1, -2) in the Tensor class
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


if MAIN:
    linear = Linear(3, 4)
    assert isinstance(linear.weight, Tensor)
    assert linear.weight.requires_grad

    input = Tensor([[1.0, 2.0, 3.0]])
    output = linear(input)
    assert output.requires_grad

    expected_output = input @ linear.weight.T + linear.bias
    np.testing.assert_allclose(output.array, expected_output.array)

    print("All tests for `Linear` passed!")


# %%
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(28 * 28, 64)
        self.linear2 = Linear(64, 64)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.output = Linear(64, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape((x.shape[0], 28 * 28))
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.output(x)
        return x


# %%

Index = Union[int, Tuple[int, ...], Tuple[Arr], Tuple[Tensor]]


def coerce_index(index: Index) -> Union[int, Tuple[int, ...], Tuple[Arr]]:
    """
    If index is of type signature `Tuple[Tensor]`, converts it to `Tuple[Arr]`.
    """
    # SOLUTION
    if isinstance(index, tuple) and set(map(type, index)) == {Tensor}:
        return tuple([i.array for i in index])
    else:
        return index


def _getitem(x: Arr, index: Index) -> Arr:
    """Like x[index] when x is a torch.Tensor."""
    # SOLUTION
    return x[coerce_index(index)]


def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    """
    Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    """
    # SOLUTION
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out


if MAIN:
    getitem = wrap_forward_fn(_getitem)
    BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)


# %%
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out


if MAIN:
    exp = wrap_forward_fn(np.exp)
    BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

    # tests.test_exp_back(Tensor)


# %%
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    """Backward function for f(x) = -x elementwise."""
    return unbroadcast(-grad_out, x)


if MAIN:
    negative = wrap_forward_fn(np.negative)
    BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

    tests.test_negative_back(Tensor)


# %%
def cross_entropy(logits: Tensor, true_labels: Tensor) -> Tensor:
    """Like torch.nn.functional.cross_entropy with reduction='none'.

    logits: shape (batch, classes)
    true_labels: shape (batch,). Each element is the index of the correct label in the logits.

    Return: shape (batch, ) containing the per-example loss.
    """
    batch_size, class_n = logits.shape
    indexed_logits = logits[arange(0, batch_size), true_labels]
    loss = -log(exp(indexed_logits) / exp(logits).sum(1))  #!!
    return loss


if MAIN:
    tests.test_cross_entropy(Tensor, cross_entropy)


# %%
class NoGrad:
    """Context manager that disables grad inside the block. Like torch.no_grad."""

    was_enabled: bool

    def __enter__(self):
        """
        Method which is called whenever the context manager is entered, i.e. at the
        start of the `with NoGrad():` block.
        """
        # SOLUTION
        global grad_tracking_enabled
        self.was_enabled = grad_tracking_enabled
        grad_tracking_enabled = False

    def __exit__(self, type, value, traceback):
        """
        Method which is called whenever we exit the context manager.
        """
        # SOLUTION
        global grad_tracking_enabled
        grad_tracking_enabled = self.was_enabled


if MAIN:
    train_loader, test_loader = get_mnist()
    visualize(train_loader)


# %%
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)


if MAIN:
    reshape = wrap_forward_fn(np.reshape)
    BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

    tests.test_reshape_back(Tensor)


# %%
class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float):
        """Vanilla SGD with no additional features."""
        self.params = list(params)
        self.lr = lr
        self.b = [None for _ in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        with NoGrad():
            for i, p in enumerate(self.params):
                assert isinstance(p.grad, Tensor)
                p.add_(p.grad, -self.lr)


def train(
    model: MLP,
    train_loader: DataLoader,
    optimizer: SGD,
    epoch: int,
    train_loss_list: Optional[list] = None,
):
    print(f"Epoch: {epoch}")
    progress_bar = tqdm(enumerate(train_loader))
    for batch_idx, (data, target) in progress_bar:
        data = Tensor(data.numpy())
        target = Tensor(target.numpy())
        optimizer.zero_grad()
        output = model(data)
        loss = cross_entropy(output, target).sum() / len(output)
        loss.backward()
        progress_bar.set_description(f"Train set: Avg loss: {loss.item():.3f}")
        optimizer.step()
        if train_loss_list is not None:
            train_loss_list.append(loss.item())


def test(model: MLP, test_loader: DataLoader, test_loss_list: Optional[list] = None):
    test_loss = 0
    correct = 0
    with NoGrad():
        for data, target in test_loader:
            data = Tensor(data.numpy())
            target = Tensor(target.numpy())
            output: Tensor = model(data)
            test_loss += cross_entropy(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.reshape(pred.shape)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"Test set:  Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({correct / len(test_loader.dataset):.1%})"
    )
    if test_loss_list is not None:
        test_loss_list.append(test_loss)


# %%
if MAIN:
    num_epochs = 5
    model = MLP()
    start = time.time()
    train_loss_list = []
    test_loss_list = []
    optimizer = SGD(model.parameters(), 0.01)
    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, epoch, train_loss_list)
        test(model, test_loader, test_loss_list)
        optimizer.step()
    print(f"\nCompleted in {time.time() - start: .2f}s")
# %%
if MAIN:
    line(
        train_loss_list,
        yaxis_range=[0, max(train_loss_list) + 0.1],
        labels={"x": "Batches seen", "y": "Cross entropy loss"},
        title="ConvNet training on MNIST",
        width=800,
        hovermode="x unified",
        template="ggplot2",  # alternative aesthetic for your plots (-:
    )
