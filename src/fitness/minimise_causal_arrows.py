from fitness.base_ff_classes.base_ff import base_ff
import re
from typing import List, Set, Tuple

class minimise_causal_arrows(base_ff):
    """
    Fitness function class for minimising the number of causal arrows in a
    probabilistic program
    """

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        p = ind.phenotype

        return count_causal_arrows(p)

V_VAR_RE = re.compile(r"\bV\d+\b")
ASSIGN_RE = re.compile(r"^\s*(?P<lhs>V\d+)\s*=\s*(?P<rhs>.+?)\s*$")
IF_RE = re.compile(r"^\s*if\b(?P<cond>.*?)(?:\{|\bthen\b)?\s*$", re.IGNORECASE)
END_IF_RE = re.compile(r"^\s*(?:end\s*if|endif|end_if|\})\s*$", re.IGNORECASE)
ELSE_RE = re.compile(r"^\s*else\b", re.IGNORECASE)


def _split_statements(program: str) -> List[str]:
    chunks: List[str] = []
    for line in program.splitlines():
        for part in line.split(";"):
            part = part.strip()
            if part:
                chunks.append(part)
    return chunks


def _vars_in_text(text: str) -> Set:
    # (Using unsubscripted Set for Python 3.7/3.8 compatibility.)
    return set(V_VAR_RE.findall(text))


def count_causal_arrows(program: str, return_edges: bool = False):
    """
    Count unique causal arrows (unique directed edges parent -> child).

    For each assignment 'Vi = RHS' inside nested ifs:
      deps = V-vars in RHS union V-vars in enclosing if-condition(s)
      for each dep in deps (excluding Vi), add edge (dep, Vi)

    Returns:
      - int (number of unique edges), OR
      - (int, set_of_edges) if return_edges=True
    """
    statements = _split_statements(program)
    cond_stack: List[Set] = []
    edges: Set[Tuple[str, str]] = set()

    for stmt in statements:
        if END_IF_RE.match(stmt):
            if cond_stack:
                cond_stack.pop()
            continue

        if ELSE_RE.match(stmt):
            continue

        m_if = IF_RE.match(stmt)
        if m_if and stmt.lstrip().lower().startswith("if"):
            cond_vars = _vars_in_text(m_if.group("cond") or "")
            cond_stack.append(cond_vars)
            continue

        m_asg = ASSIGN_RE.match(stmt)
        if m_asg:
            lhs = m_asg.group("lhs")
            rhs = m_asg.group("rhs")

            rhs_vars = _vars_in_text(rhs)

            active_cond_vars: Set = set()
            for s in cond_stack:
                active_cond_vars |= s

            deps = rhs_vars | active_cond_vars
            deps.discard(lhs)  # no self-loop

            for dep in deps:
                edges.add((dep, lhs))

    if return_edges:
        return len(edges), edges
    return len(edges)