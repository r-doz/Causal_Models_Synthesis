import numpy as np
import random
import re
from typing import List, Tuple, Optional, Dict, Set



_VKEY_RE = re.compile(r"^V(\d+)$")


def order_from_permutation(perm: Dict[str, str]) -> List[str]:
    """
    Build the candidate-program variable order from a permutation dict like:
      {'V1': 'belief', 'V2': 'skill', 'V3': 'result'}

    Returns:
      ['belief', 'skill', 'result']  (sorted by the numeric index in 'V#')
    """
    items = []
    for k, v in perm.items():
        m = _VKEY_RE.match(k.strip())
        if not m:
            raise ValueError(f"Invalid key '{k}'. Expected keys like 'V1', 'V2', ...")
        idx = int(m.group(1))
        items.append((idx, v))

    items.sort(key=lambda t: t[0])
    return [name for _, name in items]


def choose_interventions(
    data: np.ndarray,
    perm: Dict[str, str],
    max_interventions: int = 10,
    *,
    alpha: float = 0.5,
    value_sampler: str = "normal",
    normal_scale: float = 1.0,
    uniform_k: float = 2.0,
    clip_k: Optional[float] = 4.0,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[str, float]]:
    """
    Variant B, but using the *candidate program order* derived from `perm`.
    - Targets sampled without replacement with bias toward earlier vars in this order.
    - Values sampled fresh each call from empirical stats of `data`.
    
    Assumption: `data[:, i]` corresponds to the canonical order V1, V2, ...,
    and `perm` maps those to candidate variable names.
    """
    if rng is None:
        rng = np.random.default_rng()

    order = order_from_permutation(perm)

    # Empirical stats: column i corresponds to Vi -> which maps to order[i]
    eps = 1e-12
    stats = {
        order[i]: {
            "mean": float(np.mean(data[:, i])),
            "std": float(np.std(data[:, i])),
        }
        for i in range(min(len(order), data.shape[1]))
    }

    n = len(order)
    k = min(max_interventions, n)

    # Bias toward earlier variables in candidate order
    idx = np.arange(n, dtype=float)
    w = np.exp(-alpha * idx)
    w = w / w.sum()

    chosen_idx = rng.choice(n, size=k, replace=False, p=w)

    interventions: List[Tuple[str, float]] = []
    for i in chosen_idx:
        var = order[int(i)]
        m = stats[var]["mean"]
        s = max(stats[var]["std"], eps)

        if value_sampler.lower() == "normal":
            v = float(rng.normal(loc=m, scale=normal_scale * s))
        elif value_sampler.lower() == "uniform":
            lo, hi = m - uniform_k * s, m + uniform_k * s
            v = float(rng.uniform(lo, hi))
        else:
            raise ValueError(f"Unknown value_sampler='{value_sampler}'. Use 'normal' or 'uniform'.")

        if clip_k is not None:
            lo, hi = m - clip_k * s, m + clip_k * s
            v = float(np.clip(v, lo, hi))

        interventions.append((var, v))

    return interventions



def _split_statements(program: str) -> List[str]:
    """
    Split a SOGA program into individual statements.
    Supports programs written as:
      - one statement per line, OR
      - semicolon-separated statements on a single line (common in your outputs).
    """
    # Split on ';' or '\n', keep only non-empty statements
    parts = re.split(r"[;\n]+", program)
    return [p.strip() for p in parts if p.strip()]


def _is_if(stmt: str) -> bool:
    s = stmt.strip().lower()
    # adapt if you also have "if(" etc.
    return s.startswith("if ")


def _is_endif(stmt: str) -> bool:
    s = stmt.strip().lower()
    # support both "endif" and "end if"
    return s == "endif" or s == "end if" or s.startswith("endif") or s.startswith("end if")


def apply_intervention_to_program(program: str, var: str, value) -> str:
    """
    Apply do(var = value) to a SOGA program.

    - Remove any assignment to var (e.g. "var = ...")
    - Remove any IF block whose body contains an assignment to var
      (regardless of the IF condition)
    - Keep IF blocks that only *depend on var* (children logic)
    - Insert "var = value" at the top

    IMPORTANT: Works for semicolon-separated single-line programs too.
    """

    stmts = _split_statements(program)

    # Match "var = ..." with flexible whitespace
    assign_re = re.compile(rf"^\s*{re.escape(var)}\s*=")

    remove_idx: Set[int] = set()
    stack = []  # each entry: [if_start_index, has_assignment_to_var_inside]

    # Pass 1: detect assignments and mark IF blocks to remove
    for i, stmt in enumerate(stmts):
        s = stmt.strip()

        if _is_if(s):
            stack.append([i, False])
            continue

        if _is_endif(s):
            if stack:
                start_i, contains_var = stack.pop()
                if contains_var:
                    for j in range(start_i, i + 1):
                        remove_idx.add(j)
            # If stack is empty, it's a malformed program; we just ignore.
            continue

        # Direct assignment to var → remove this statement
        if assign_re.match(s):
            remove_idx.add(i)
            # If inside an IF, mark that IF block for removal
            if stack:
                stack[-1][1] = True

    # Pass 2: keep only non-removed statements
    kept = [stmt for i, stmt in enumerate(stmts) if i not in remove_idx]

    # Prepend intervention assignment
    intervention_stmt = f"{var} = {value}"
    kept = [intervention_stmt] + kept

    # Re-join in the same “semicolon program” style you use
    return ";".join(kept) + ";"
