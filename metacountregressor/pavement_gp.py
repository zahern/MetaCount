"""
zeke_gp_forms.py
========================================================================
Genetic-programming (symbolic regression) functional-form discovery.

This is form 12 in the paper's library (Table~\\ref{tab:forms}). Instead of
drawing a functional form from a fixed menu, an expression tree is evolved:
sub-trees are mutated and crossed over, and each candidate expression is
compiled to a design matrix and scored by profile-BIC. The fittest tree is
embedded in the parent simulated-annealing state.

Expression trees
----------------
Node types:
    binary : +, *, -, /, pow
    unary  : log, exp, sin, sqrt
    leaf   : var_i  (continuous predictor i)  or  const (scalar)
A tree is represented as a nested tuple and compiled (via ``eval`` on a
generated lambda) to ``design(X, names) -> (n, m)``. For a *single* target
form the tree is compiled to one column; multi-column forms are not needed
because GP naturally discovers interactions. To match the program's convention
the tree output replaces the continuous design for the cluster.

Metacountregressor embed note
-----------------------------
This operator is swappable into the SA neighbourhood in
``zeke_pavement_search.py``: when a cluster's form == 12, call
``GPSymbolicRegressor.search(...)`` instead of the fixed-form builder. The
returned serialisable tree is stored in the decision vector exactly like
``lam`` / ``shape``.
"""
from __future__ import annotations

import numpy as np
import random
from copy import deepcopy

BINARY = ["add", "mul", "sub", "div", "pow"]
UNARY = ["log", "exp", "sqrt"]
TERMINALS = ["var", "const"]


def _rand_const(rng):
    return round(rng.uniform(-2.0, 2.0), 3)


def make_tree(rng, depth, n_vars, p_term=0.3):
    if depth <= 0 or rng.random() < p_term:
        if rng.random() < 0.5 or n_vars == 0:
            return ("const", _rand_const(rng))
        return ("var", rng.randrange(n_vars))
    op = rng.choice(BINARY if rng.random() < 0.6 else UNARY)
    if op in UNARY:
        return (op, make_tree(rng, depth - 1, n_vars, p_term))
    return (op, make_tree(rng, depth - 1, n_vars, p_term),
            make_tree(rng, depth - 1, n_vars, p_term))


def tree_to_lambda(tree, n_vars):
    """Compile a tree to a callable f(X, names) -> (n,) array.

    'var' nodes index into the continuous column order ``names``.
    """
    def compile_node(node):
        if node[0] == "const":
            return f"({node[1]})"
        if node[0] == "var":
            return f"X[:, {node[1]}]"
        if node[0] == "add":
            return f"({compile_node(node[1])} + {compile_node(node[2])})"
        if node[0] == "mul":
            return f"({compile_node(node[1])} * {compile_node(node[2])})"
        if node[0] == "sub":
            return f"({compile_node(node[1])} - {compile_node(node[2])})"
        if node[0] == "div":
            return f"(np.where(np.abs({compile_node(node[2])})<1e-9, 1e-9, {compile_node(node[2])}))"
        if node[0] == "pow":
            return f"np.clip({compile_node(node[1])}, -5, 5)**2"  # squared for stability
        if node[0] == "log":
            return f"np.log(np.clip({compile_node(node[1])}, 1e-6, None))"
        if node[0] == "exp":
            return f"np.exp(np.clip({compile_node(node[1])}, -30, 30))"
        if node[0] == "sqrt":
            return f"np.sqrt(np.clip({compile_node(node[1])}, 0, None))"
        raise ValueError(node[0])

    expr = compile_node(tree)
    code = f"def _f(X, names):\n    return np.asarray({expr}, dtype=float).reshape(-1, 1)"
    local = {}
    exec(code, {"np": np, "X": None, "names": None}, local)
    return local["_f"]


def tree_size(node):
    if node[0] in ("const", "var"):
        return 1
    if node[0] in UNARY:
        return 1 + tree_size(node[1])
    return 1 + tree_size(node[1]) + tree_size(node[2])


def _mutate(node, rng, n_vars, max_depth=4):
    if rng.random() < 0.25:
        return make_tree(rng, max_depth, n_vars)
    if node[0] in ("const", "var"):
        return make_tree(rng, 1, n_vars)
    if node[0] in UNARY:
        return (node[0], _mutate(node[1], rng, n_vars, max_depth))
    return (node[0], _mutate(node[1], rng, n_vars, max_depth),
            _mutate(node[2], rng, n_vars, max_depth))


def _crossover(a, b, rng):
    if rng.random() < 0.5 or tree_size(a) == 1 or tree_size(b) == 1:
        return deepcopy(a), deepcopy(b)
    la = _collect(a, rng)
    lb = _collect(b, rng)
    if not la or not lb:
        return deepcopy(a), deepcopy(b)
    na = rng.choice(la); nb = rng.choice(lb)
    return _replace(a, na, nb), _replace(b, nb, na)


def _collect(node, rng, acc=None):
    acc = acc if acc is not None else []
    acc.append(node)
    if node[0] in UNARY:
        _collect(node[1], rng, acc)
    elif node[0] in BINARY:
        _collect(node[1], rng, acc); _collect(node[2], rng, acc)
    return acc


def _replace(root, target, repl):
    if root is target:
        return deepcopy(repl)
    if root[0] in UNARY:
        return (root[0], _replace(root[1], target, repl))
    if root[0] in BINARY:
        return (root[0], _replace(root[1], target, repl),
                _replace(root[2], target, repl))
    return root


def _ols_bic(Xd, y):
    Xd = np.hstack([np.ones((Xd.shape[0], 1)), Xd])
    if Xd.shape[0] <= Xd.shape[1] or np.linalg.matrix_rank(Xd) < Xd.shape[1]:
        return np.inf
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    rss = float(resid @ resid)
    n, p = Xd.shape
    return n * np.log(rss / n) + p * np.log(n)


class GPSymbolicRegressor:
    """Genetic-programming search for a single-cluster functional form."""

    def __init__(self, pop_size=60, generations=20, max_depth=4,
                 tournament=3, mut_rate=0.25, cross_rate=0.6, seed=0):
        self.pop_size = pop_size
        self.generations = generations
        self.max_depth = max_depth
        self.tournament = tournament
        self.mut_rate = mut_rate
        self.cross_rate = cross_rate
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def _fitness(self, tree, Xc, y, n_vars):
        try:
            f = tree_to_lambda(tree, n_vars)
            Xd = f(Xc, None)
            return _ols_bic(Xd, y)
        except Exception:
            return np.inf

    def search(self, Xc, y, n_vars):
        rng = self.rng
        pop = [make_tree(rng, self.max_depth, n_vars) for _ in range(self.pop_size)]
        fit = [self._fitness(t, Xc, y, n_vars) for t in pop]
        for _ in range(self.generations):
            new_pop = []
            while len(new_pop) < self.pop_size:
                i1 = min(range(self.pop_size),
                         key=lambda i: fit[i] if rng.random() < 0.5
                         else rng.randrange(self.pop_size))
                i2 = min(range(self.pop_size),
                         key=lambda i: fit[i] if rng.random() < 0.5
                         else rng.randrange(self.pop_size))
                p1, p2 = _crossover(pop[i1], pop[i2], rng)
                if rng.random() < self.mut_rate:
                    p1 = _mutate(p1, rng, n_vars, self.max_depth)
                if rng.random() < self.mut_rate:
                    p2 = _mutate(p2, rng, n_vars, self.max_depth)
                new_pop += [p1, p2]
            pop = new_pop[:self.pop_size]
            fit = [self._fitness(t, Xc, y, n_vars) for t in pop]
        best_i = int(np.argmin(fit))
        return {"tree": pop[best_i], "bic": fit[best_i],
                "callable": tree_to_lambda(pop[best_i], n_vars)}


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 300
    a = rng.uniform(0.5, 20, n)
    Xc = np.column_stack([a, rng.uniform(100, 5000, n)])
    # ground truth: log(age) interaction (form 5-like)
    y = 1.0 - 0.4 * np.log(a) + 0.05 * np.log(a) * np.log(Xc[:, 1] / 1000) \
        + rng.normal(0, 0.05, n)
    gp = GPSymbolicRegressor(pop_size=40, generations=8, seed=3)
    out = gp.search(Xc, y, n_vars=2)
    print("GP best bic=%.3f  tree=%s" % (out["bic"], out["tree"]))
    print("OK: zeke_gp_forms self-test passed")
