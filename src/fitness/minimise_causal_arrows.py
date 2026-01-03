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

# Matches a "simple linear term" (no extra operators beyond optional scalar*Vn):
#   V2
#   3*V2
#   3.5 * V2
_SIMPLE_LINEAR_TERM_RE = re.compile(
    r"^\s*(?:(?P<coef>\d+(?:\.\d+)?)\s*\*\s*)?(?P<var>V\d+)\s*$"
)


def _split_statements(program: str) -> List[str]:
    chunks: List[str] = []
    for line in program.splitlines():
        for part in line.split(";"):
            part = part.strip()
            if part:
                chunks.append(part)
    return chunks


def _vars_in_text(text: str) -> Set:
    """Extract all V-variables appearing in text (used for IF conditions)."""
    return set(V_VAR_RE.findall(text))


def _split_top_level_additive(expr: str) -> List[Tuple[int, str]]:
    """
    Split an expression into top-level additive terms while respecting nesting.

    Returns a list of (sign, term) where sign is +1 or -1.
    Only splits on '+'/'-' when they occur at top-level (not inside (), [], {}).
    """
    terms: List[Tuple[int, str]] = []
    depth_paren = depth_brack = depth_brace = 0

    i = 0
    n = len(expr)

    # Current term accumulation
    cur = []
    sign = +1  # sign for the current term

    # Allow a leading unary +/-
    while i < n and expr[i].isspace():
        i += 1
    if i < n and expr[i] in "+-":
        sign = +1 if expr[i] == "+" else -1
        i += 1

    while i < n:
        ch = expr[i]

        # Track nesting so we don't split inside function calls / lists / parentheses
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)

        # Split on + / - only at top level
        if ch in "+-" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            term_str = "".join(cur).strip()
            if term_str:
                terms.append((sign, term_str))
            cur = []
            sign = +1 if ch == "+" else -1
            i += 1
            continue

        cur.append(ch)
        i += 1

    # Last term
    term_str = "".join(cur).strip()
    if term_str:
        terms.append((sign, term_str))

    return terms


def _vars_in_rhs_with_linear_cancellation(rhs: str) -> Set:
    """
    Extract RHS dependencies, but drop variables that cancel out purely via
    top-level additive simple-linear terms, e.g., V1 - V1, 2*V1 - 2*V1.

    Safety rule:
      - We only cancel contributions coming from *simple linear terms* at top level.
      - If a variable appears anywhere in a non-simple term, it is kept.
    """
    # Quick path: no variables
    all_vars = set(V_VAR_RE.findall(rhs))
    if not all_vars:
        return set()

    terms = _split_top_level_additive(rhs)

    # Sum coefficients for vars that appear as simple linear terms
    coef_sum = {}  # var -> float
    vars_in_simple_terms = set()
    vars_in_complex_terms = set()

    for sgn, term in terms:
        # Check if the whole term is a simple linear term (optional scalar * Vn)
        m = _SIMPLE_LINEAR_TERM_RE.match(term)
        if m:
            var = m.group("var")
            coef = m.group("coef")
            c = float(coef) if coef is not None else 1.0
            coef_sum[var] = coef_sum.get(var, 0.0) + sgn * c
            vars_in_simple_terms.add(var)
        else:
            # Any variables inside this term are "complex occurrences"
            vars_in_complex_terms |= set(V_VAR_RE.findall(term))

    # Start by keeping everything
    deps = set(all_vars)

    # Cancel a var only if:
    #  (i) it appears in at least one simple term,
    # (ii) its simple-term coefficient sum is zero,
    # (iii) it does NOT appear in any complex term.
    for var in vars_in_simple_terms:
        if abs(coef_sum.get(var, 0.0)) < 1e-12 and var not in vars_in_complex_terms:
            deps.discard(var)

    return deps


def count_causal_arrows(program: str, return_edges: bool = False):
    """
    Count unique causal arrows (unique directed edges parent -> child).

    For each assignment 'Vi = RHS' inside nested ifs:
      deps = V-vars in RHS (with simple linear cancellation)
             union V-vars in enclosing if-condition(s)
      for each dep in deps (excluding Vi), add edge (dep, Vi)
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

            rhs_vars = _vars_in_rhs_with_linear_cancellation(rhs)

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
