from fitness.base_ff_classes.base_ff import base_ff
import re
from typing import List, Set, Tuple, Iterable, FrozenSet
from algorithm.parameters import params
from os import path

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

        num, edges =count_causal_arrows(p)

        if params['SAVE_STRUCTURES']:
            structure_id = structure_to_id(edges)
            ind.structure_id = structure_id

        return num


V_VAR_RE = re.compile(r"\bV\d+\b")
ASSIGN_RE = re.compile(r"^\s*(?P<lhs>V\d+)\s*=\s*(?P<rhs>.+?)\s*$")
IF_RE = re.compile(r"^\s*if\b(?P<cond>.*?)(?:\{|\bthen\b)?\s*$", re.IGNORECASE)
END_IF_RE = re.compile(r"^\s*(?:end\s*if|endif|end_if|\})\s*$", re.IGNORECASE)
ELSE_RE = re.compile(r"^\s*else\b", re.IGNORECASE)

# Scientific notation-friendly number
_NUM_RE = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

# Matches a "simple linear term" at top level, allowing explicit unary signs:
#   +V2
#   -V2
#   +3*+V2
#   -2.5e-3 * -V2
_SIMPLE_LINEAR_TERM_RE = re.compile(
    rf"^\s*(?P<usign>[+-])?\s*"
    rf"(?:(?P<coef>{_NUM_RE})\s*\*\s*)?"
    rf"(?P<vsign>[+-])?\s*(?P<var>V\d+)\s*$"
)


def _split_statements(program: str) -> List[str]:
    chunks: List[str] = []
    for line in program.splitlines():
        for part in line.split(";"):
            part = part.strip()
            if part:
                chunks.append(part)
    return chunks


def _vars_in_text(text: str) -> Set[str]:
    """Extract all V-variables appearing in text (used for IF conditions)."""
    return set(V_VAR_RE.findall(text))


def _split_top_level_additive(expr: str) -> List[Tuple[int, str]]:
    """
    Split an expression into top-level additive terms while respecting nesting.

    Returns a list of (sign, term) where sign is +1 or -1.

    IMPORTANT:
      - Only splits on *binary* '+'/'-' at top level.
      - Unary '+'/'-' (e.g., '+V1', '-V2', '+3*+V1') is kept inside the term.
    """
    terms: List[Tuple[int, str]] = []
    depth_paren = depth_brack = depth_brace = 0

    i = 0
    n = len(expr)

    cur: List[str] = []
    sign = +1  # sign for current term due to binary splitting

    def _last_nonspace_char(buf: List[str]) -> str | None:
        for ch in reversed(buf):
            if not ch.isspace():
                return ch
        return None

    # Consume leading spaces
    while i < n and expr[i].isspace():
        i += 1

    # Leading unary sign for the whole expression becomes the initial "sign"
    if i < n and expr[i] in "+-":
        sign = +1 if expr[i] == "+" else -1
        i += 1

    while i < n:
        ch = expr[i]

        # Track nesting so we don't split inside () [] {}
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

        # Candidate split on +/-
        if ch in "+-" and depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            # Decide unary vs binary:
            # Unary if current term buffer is empty (ignoring spaces),
            # or if it follows another operator or an opening bracket.
            last = _last_nonspace_char(cur)

            is_unary = (last is None) or (last in "+-*/(,[{")

            if not is_unary:
                # Binary operator => split
                term_str = "".join(cur).strip()
                if term_str:
                    terms.append((sign, term_str))
                cur = []
                sign = +1 if ch == "+" else -1
                i += 1
                continue

            # Unary => keep inside current term
            cur.append(ch)
            i += 1
            continue

        cur.append(ch)
        i += 1

    term_str = "".join(cur).strip()
    if term_str:
        terms.append((sign, term_str))

    return terms


def _normalize_top_level_term(term: str) -> str:
    """
    Conservative normalization for exact term cancellation:
      - remove all whitespace
      - remove unary '+' signs (keep '-')
    This makes '+V1*+V1' normalize to 'V1*V1', and ' + V1 ' to 'V1'.

    NOTE: We do NOT do algebraic normalization (no reordering, no simplifying).
    """
    t = re.sub(r"\s+", "", term)
    t = t.replace("+", "")  # remove explicit unary plus everywhere
    return t


def _vars_in_rhs_with_linear_cancellation(rhs: str) -> Set[str]:
    """
    Extract RHS dependencies, but drop variables that cancel out via:
      (A) top-level additive simple-linear cancellation (your original rule), AND
      (B) exact cancellation of identical top-level complex terms: T - T.

    Safety / conservativeness:
      - We only cancel complex terms if they are EXACTLY identical after light normalization.
      - No commutativity/associativity reasoning (e.g., V1*V2 vs V2*V1 are NOT considered equal).
    """
    all_vars = set(V_VAR_RE.findall(rhs))
    if not all_vars:
        return set()

    terms = _split_top_level_additive(rhs)

    # ---- Part A: simple linear term coefficient sum (same as before, but sign-aware) ----
    linear_coef_sum: dict[str, float] = {}
    vars_in_simple_terms: Set[str] = set()

    # ---- Part B: exact top-level term cancellation for complex terms ----
    # We sum coefficients (+1/-1) for each normalized term string.
    complex_term_sum: dict[str, float] = {}
    complex_term_vars: dict[str, Set[str]] = {}

    for bin_sgn, term in terms:
        m = _SIMPLE_LINEAR_TERM_RE.match(term)
        if m:
            var = m.group("var")

            us = -1.0 if (m.group("usign") == "-") else 1.0
            vs = -1.0 if (m.group("vsign") == "-") else 1.0

            coef_str = m.group("coef")
            coef = float(coef_str) if coef_str is not None else 1.0

            total = float(bin_sgn) * us * coef * vs
            linear_coef_sum[var] = linear_coef_sum.get(var, 0.0) + total
            vars_in_simple_terms.add(var)
        else:
            norm = _normalize_top_level_term(term)
            complex_term_sum[norm] = complex_term_sum.get(norm, 0.0) + float(bin_sgn)
            complex_term_vars.setdefault(norm, set()).update(V_VAR_RE.findall(term))

    # Vars that survive due to simple-linear part
    vars_from_linear = {v for v, s in linear_coef_sum.items() if abs(s) >= 1e-12}

    # Vars that survive due to complex terms that DO NOT cancel
    vars_from_complex_survivors: Set[str] = set()
    for norm, s in complex_term_sum.items():
        if abs(s) >= 1e-12:
            vars_from_complex_survivors |= complex_term_vars.get(norm, set())

    # Final dependencies: only vars that survive either component
    return vars_from_linear | vars_from_complex_survivors

def count_causal_arrows(program: str, return_edges: bool = False):
    """
    Count unique causal arrows (unique directed edges parent -> child).

    For each assignment 'Vi = RHS' inside nested ifs:
      deps = V-vars in RHS (with simple linear cancellation)
             union V-vars in enclosing if-condition(s)
      for each dep in deps (excluding Vi), add edge (dep, Vi)
    """
    statements = _split_statements(program)
    cond_stack: List[Set[str]] = []
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

            active_cond_vars: Set[str] = set()
            for s in cond_stack:
                active_cond_vars |= s

            deps = rhs_vars | active_cond_vars
            deps.discard(lhs)  # no self-loop

            for dep in deps:
                edges.add((dep, lhs))

    return len(edges), edges

def structure_to_id(edge_set):
    """
    edge_set: iterable of ('Vi','Vj') tuples
    returns: integer ID in [1,27]
    """
    key = frozenset(edge_set)
    try:
        return STRUCTURE_TO_ID[key]
    except KeyError:
        raise ValueError(f"Invalid causal structure: {edge_set}")


# ============================================================
# Manual mapping: frozenset({(Vi, Vj), ...}) -> contiguous ID
# Grouped by number of edges
# ============================================================

STRUCTURE_TO_ID = {

    # --------------------------------------------------------
    # 0 EDGES (1 structure)
    # --------------------------------------------------------
    frozenset(): 1,

    # --------------------------------------------------------
    # 1 EDGE (6 structures) -> IDs 2–7
    # --------------------------------------------------------
    frozenset({('V1', 'V2')}): 2,
    frozenset({('V2', 'V1')}): 3,
    frozenset({('V1', 'V3')}): 4,
    frozenset({('V3', 'V1')}): 5,
    frozenset({('V2', 'V3')}): 6,
    frozenset({('V3', 'V2')}): 7,

    # --------------------------------------------------------
    # 2 EDGES (12 structures) -> IDs 8–19
    # --------------------------------------------------------
    frozenset({('V1', 'V2'), ('V1', 'V3')}): 8,
    frozenset({('V1', 'V2'), ('V3', 'V1')}): 9,
    frozenset({('V2', 'V1'), ('V1', 'V3')}): 10,
    frozenset({('V2', 'V1'), ('V3', 'V1')}): 11,

    frozenset({('V1', 'V2'), ('V2', 'V3')}): 12,
    frozenset({('V1', 'V2'), ('V3', 'V2')}): 13,
    frozenset({('V2', 'V1'), ('V2', 'V3')}): 14,
    frozenset({('V2', 'V1'), ('V3', 'V2')}): 15,

    frozenset({('V1', 'V3'), ('V2', 'V3')}): 16,
    frozenset({('V1', 'V3'), ('V3', 'V2')}): 17,
    frozenset({('V3', 'V1'), ('V2', 'V3')}): 18,
    frozenset({('V3', 'V1'), ('V3', 'V2')}): 19,

    # --------------------------------------------------------
    # 3 EDGES (8 structures) -> IDs 20–27
    # --------------------------------------------------------
    frozenset({('V1', 'V2'), ('V1', 'V3'), ('V2', 'V3')}): 20,
    frozenset({('V1', 'V2'), ('V1', 'V3'), ('V3', 'V2')}): 21,
    frozenset({('V1', 'V2'), ('V3', 'V1'), ('V2', 'V3')}): 22,
    frozenset({('V1', 'V2'), ('V3', 'V1'), ('V3', 'V2')}): 23,

    frozenset({('V2', 'V1'), ('V1', 'V3'), ('V2', 'V3')}): 24,
    frozenset({('V2', 'V1'), ('V1', 'V3'), ('V3', 'V2')}): 25,
    frozenset({('V2', 'V1'), ('V3', 'V1'), ('V2', 'V3')}): 26,
    frozenset({('V2', 'V1'), ('V3', 'V1'), ('V3', 'V2')}): 27,
}



Edge = Tuple[str, str]

# ============================================================
# Undirected (non-oriented) structures for 3 vars:
# possible undirected edges are {V1-V2, V1-V3, V2-V3}
# -> 2^3 = 8 structures total
#
# IDs are contiguous by number of undirected edges:
#  1: 0 edges
#  2-4: 1 edge
#  5-7: 2 edges
#  8: 3 edges
# ============================================================

def _undirected_key(edge_set: Iterable[Edge]) -> FrozenSet[FrozenSet[str]]:
    """
    Canonicalize a directed edge set into an undirected structure key.

    Example:
      {('V2','V1'), ('V1','V3')} -> frozenset({frozenset({'V1','V2'}),
                                              frozenset({'V1','V3'})})
    """
    undirected_edges = set()
    for a, b in edge_set:
        if a == b:
            # ignore self loops (shouldn't exist anyway)
            continue
        undirected_edges.add(frozenset((a, b)))  # drops orientation
    return frozenset(undirected_edges)


# Manual mapping: undirected-edge frozensets -> ID in [1,8]
STRUCTURE_TO_ID_UNDIRECTED = {
    # 0 edges
    frozenset(): 1,

    # 1 edge (IDs 2–4)
    frozenset({frozenset(("V1", "V2"))}): 2,
    frozenset({frozenset(("V1", "V3"))}): 3,
    frozenset({frozenset(("V2", "V3"))}): 4,

    # 2 edges (IDs 5–7)
    frozenset({frozenset(("V1", "V2")), frozenset(("V1", "V3"))}): 5,
    frozenset({frozenset(("V1", "V2")), frozenset(("V2", "V3"))}): 6,
    frozenset({frozenset(("V1", "V3")), frozenset(("V2", "V3"))}): 7,

    # 3 edges (triangle)
    frozenset({frozenset(("V1", "V2")), frozenset(("V1", "V3")), frozenset(("V2", "V3"))}): 8,
}


def structure_to_id_undirected(edge_set: Iterable[Edge]) -> int:
    """
    edge_set: iterable of directed edges, e.g. {('V2','V1'), ('V1','V3')}
    returns: undirected-structure ID in [1,8]
    """
    key = _undirected_key(edge_set)
    try:
        return STRUCTURE_TO_ID_UNDIRECTED[key]
    except KeyError:
        raise ValueError(f"Invalid undirected causal structure key={key} from edge_set={set(edge_set)}")