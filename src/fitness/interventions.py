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
    Apply do(var = value) to a SOGA if F < 8.5 {;T = F;} else {;T = 20;} end if;program.

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
    kept = remove_empty_if_blocks_tokens(kept)
    # Re-join in the same “semicolon program” style you use
    out = []
    for stmt in kept:
        stmt = stmt.strip()
        if not stmt:
            continue

        if _is_control_statement(stmt):
            out.append(stmt)
        else:
            out.append(stmt + ";")
    
    return "\n".join(out)



def _is_control_statement(stmt: str) -> bool:
    s = stmt.strip().lower()
    return (
        s.startswith("if ")
        or s == "else"
        or s.startswith("else ")
        or s.endswith("{")
        or s == "}"
        or s.startswith("end if")
    )

from typing import List

def remove_empty_if_blocks_tokens(tokens: List[str]) -> List[str]:
    """
    Remove IF blocks that became empty after intervention removal.

    Expected token patterns (your case):
      - IF start token:        "if ... {"
      - ELSE marker token:     "} else {"
      - END marker token:      "} end if;"   (or "} endif;")

    Works with nested IFs (stack-based). If an inner IF is removed, outer IF emptiness
    is evaluated after that, so nested cleanup works in one pass.
    """

    def is_if(tok: str) -> bool:
        return tok.strip().lower().startswith("if ")

    def is_else(tok: str) -> bool:
        return tok.strip().lower().startswith("} else")

    def is_end_if(tok: str) -> bool:
        s = tok.strip().lower()
        return s.startswith("} end if") or s.startswith("} endif")

    changed = True
    cur = tokens

    # Loop-to-stability (cheap + safe)
    while changed:
        changed = False
        out: List[str] = []

        # frame: dict with keys: if_tok, then, else, mode, else_tok, end_tok
        stack = []

        def append_to_current(tok: str):
            if stack:
                frame = stack[-1]
                if frame["mode"] == "then":
                    frame["then"].append(tok)
                else:
                    frame["else"].append(tok)
            else:
                out.append(tok)

        for tok in cur:
            t = tok.strip()

            if is_if(t):
                stack.append({
                    "if_tok": tok,
                    "then": [],
                    "else": [],
                    "mode": "then",
                    "else_tok": None,
                    "end_tok": None
                })
                continue

            if stack and is_else(t):
                stack[-1]["mode"] = "else"
                stack[-1]["else_tok"] = tok
                continue

            if stack and is_end_if(t):
                frame = stack.pop()
                frame["end_tok"] = tok

                # IF is empty if both branches have no tokens
                if len(frame["then"]) == 0 and len(frame["else"]) == 0:
                    changed = True
                    # drop the whole if/else/end-if structure
                    continue

                # otherwise, reconstruct the block exactly as tokenized
                rebuilt = [frame["if_tok"]] + frame["then"]

                # In your grammar, else exists; keep it if present
                if frame["else_tok"] is not None:
                    rebuilt += [frame["else_tok"]] + frame["else"]

                rebuilt += [frame["end_tok"]]

                # append rebuilt block to parent or output
                for rt in rebuilt:
                    append_to_current(rt)

                continue

            # normal token
            append_to_current(tok)

        # If malformed program (unclosed if), just flush frames (conservatively keep)
        while stack:
            frame = stack.pop(0)
            # keep what we saw, rather than deleting anything
            out.append(frame["if_tok"])
            out.extend(frame["then"])
            if frame["else_tok"] is not None:
                out.append(frame["else_tok"])
                out.extend(frame["else"])

        cur = out

    return cur
