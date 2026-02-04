from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
#from stats.stats import stats
from scipy.stats import multivariate_normal
import random
import time as timeit
import signal
import sys
import numpy as np
import re
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import fitness.data_generating_process as dgp
import threading
from math import isfinite
import fitness.interventions as interventions
from os import path

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../SOGA-main/src')
from sogaPreprocessor import *
from producecfg import *
from libSOGA import *

torch.set_default_dtype(torch.float64)
# Define a custom exception for timeouts
class TimeoutException(Exception):
    pass

def timeout_handler():
    raise TimeoutException()

# Define a handler function for the timeout
#def handler(signum, frame):
#    raise TimeoutException("Code execution exceeded time limit")

def compute_likelihood(p, data_var_list, data):
    """ computes the likelihood of output_dist with respect to variables data_var_list sampled in data """

    time_start_execution = timeit.time()
    # Computes output distribution of the program
    try:
        compiledText=compile2SOGA_text(p)
        cfg = produce_cfg_text(compiledText)
    except Exception as e:
        print(p)
    try:                                
        output_dist = start_SOGA(cfg)
    except IndexError: # program has no valid paths
        #stats['invalids'] += 1
        print("Program has no valid paths")
        return torch.tensor(-1e6)
    time_end_execution = timeit.time()
    #print(f"Time to execute program: {time_end_execution - time_start_execution} seconds")
    time_comptation_start = timeit.time()
    data = torch.tensor(data)
    likelihood = 0
    # extract indexes of the variables in the data
    try:
        data_var_index = [output_dist.var_list.index(element) for element in data_var_list ]
    except ValueError:  # if the program doesn't have all the variables we are using for the likelihood
            print("Program missing data variables")
            return torch.tensor(-1e6)
    except:
            raise
    for k in range(output_dist.gm.n_comp()):
        # extract the covariance matrix only for the variables in the data
        sigma = output_dist.gm.sigma[k][data_var_index][:, data_var_index]
        # first I consider the mu only for variables in the data
        mu = torch.tensor(output_dist.gm.mu[k][data_var_index])
        # selects indices of delta (discrete) variables and non-delta (continuous) variables
        deltas = np.where(np.diag(sigma) == 0)[0]
        not_deltas = np.where(np.diag(sigma) != 0)[0]
        # saves means of delta and non-delta variables and covariance matrix of non-delta
        mu_delta = mu[deltas]
        mu_not_delta = mu[not_deltas]
        sigma_not_delta = torch.tensor(sigma[not_deltas][:, not_deltas])
        # computes pdf of non-delta variables 
        if len(mu_not_delta) >= 1:  # if there is at least one continuous variable
            continuous_pdf = output_dist.gm.pi[k]*MultivariateNormal(mu_not_delta, sigma_not_delta).log_prob(data[:,not_deltas]).exp()
        else:
            continuous_pdf = output_dist.gm.pi[k]*torch.ones(len(data))
        # computes pmf of delta variables
        if len(mu_delta) >= 1:   # if there is at least one discrete variable
            discrete_pmf = torch.all((mu_delta == data[:, deltas]),dim=1)
        else:
            discrete_pmf = torch.ones(len(data))
        #except ValueError:  # if the covariance matrix is singular
        #    return torch.tensor(-np.inf)
        #except:
        #    raise
        likelihood += continuous_pdf*discrete_pmf # sums likelihood of every data over all components
    time_comptation_end = timeit.time()
    #print(f"Time to compute likelihood: {time_comptation_end - time_comptation_start} seconds")
    if torch.sum(torch.log(likelihood))/len(data) < torch.tensor(-1e6):
        #print("Likelihood too low")
        return torch.tensor(-1e6)

    return torch.sum(torch.log(likelihood))/len(data)

class soga_fitness_SCM(base_ff):
    """Fitness function for finding the length of the shortest path between
    two nodes in a grade compared to the known shortest path. Takes a path (
    list of node labels as strings) and returns fitness. Penalises output
    that is not the same length as the target."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def evaluate(self, ind, **kwargs):
        self.default_fitness = torch.tensor(-1e6)
        p = ind.phenotype
        #p = smooth_program(p)
        #print("\n" + p)
        #print("\n -----------------------------------------")

        fitness = 0
        #timer = threading.Timer(10, timeout_handler)
        try:
            #timer.start()
            fitness, processed_program = likelihood_of_program_wrt_data(p)
        except TimeoutException as e:
            #print("Caught TimeoutException")
            fitness = self.default_fitness
        except:
            #print("Caught general SOGA exception")
            fitness = self.default_fitness
            #I do not define the indiviaduals as invalid in order to allow crossover
            #if not hasattr(params['FITNESS_FUNCTION'], "multi_objective"):
                #stats['invalids'] += 1
        #finally:
            #timer.cancel()
        
    
        return fitness

def generate_list():
    return [random.randint(0, round(random.random() * 90 + 10)) for i in range(9)]

def preprocess_program(p):
    p = convert_numbers_to_floats(p)
    p = pre_process_instructions(p)
    p = convert_and_normalize_gm_structure(p)
    p = convert_uniform_structure(p)
    return p


def convert_and_normalize_gm_structure(text):
    # Regular expression to find gm structure
    pattern = r'gm\(\s*(\[[^\]]+\](?:,\s*\[[^\]]+\])*)\s*\)'
    
    # Match all occurrences of the structure
    matches = re.findall(pattern, text)
    
    # Process each match
    converted_text = text
    for match in matches:
        # Find all sets of [pi, mu, s] inside the matched string
        elements = re.findall(r'\[\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\]', match)
        
        # Separate pi, mu, and s into their own lists
        pi_list = [float(e[0]) for e in elements]
        mu_list = [e[1] for e in elements]
        s_list = [e[2] for e in elements]
        #print(s_list)
        
        # Normalize pi_list
        pi_sum = sum(pi_list)
        normalized_pi_list = [pi / pi_sum for pi in pi_list] if pi_sum != 0 else [pi + 1. for pi in pi_list]
        
        # Format the new gm structure with normalized pi_list
        new_gm = f'gm([{", ".join(f"{pi:.6f}" for pi in normalized_pi_list)}], [{", ".join(mu_list)}], [{", ".join(s_list)}])'
        
        # Replace the old structure with the new one in the text
        converted_text = converted_text.replace(f'gm({match})', new_gm)
    
    return converted_text

def convert_uniform_structure(text):
    # Regular expression to find the structure uniform([a, b], c)
    pattern = r'uniform\(\s*\[\s*([0-9.-]+)\s*,\s*([0-9.-]+)\s*\]\s*,\s*([0-9.-]+)\s*\)'
    
    # Find all matches of uniform([a, b], c)
    matches = re.findall(pattern, text)
    
    # Process each match
    converted_text = text
    for match in matches:
        a = float(match[0])  # Extract 'a'
        b = float(match[1])  # Extract 'b'
        c = match[2]         # Extract 'c'
        
        # New value for 'a + b'
        new_b = a + b
        
        # Format the new uniform structure
        new_uniform = f'uniform([{a:.6f}, {new_b:.6f}], 2)'
        
        # Replace the old structure with the new one in the text
        old_uniform = f'uniform([{match[0]}, {match[1]}], {match[2]})'
        converted_text = converted_text.replace(old_uniform, new_uniform)
    
    return converted_text

def extract_perm_and_strip_program(program_str: str):
    """
    Extract the permutation integer from the first non-empty line
    and return:
       perm: int
       program_without_perm: str
    """
    lines = [line.rstrip() for line in program_str.split("\n")]

    # Find the first non-empty line → perm
    for i, line in enumerate(lines):
        if line.strip():
            perm = int(line.strip())
            # Return program without that line
            program_without_perm = "\n".join(lines[i+1:])
            return perm, program_without_perm

    raise ValueError("Program is empty or has no valid perm line.")

import itertools

def perm_to_mapping(perm: int, dataset_vars):
    """
    Given perm (1..N!) and a list of dataset variable names,
    return mapping Vi -> dataset_vars according to the permutation index.
    """

    N = len(dataset_vars)

    # All permutations in lexicographic order
    all_perms = list(itertools.permutations(range(N)))

    if perm < 1 or perm > len(all_perms):
        raise ValueError(f"Invalid perm: must be between 1 and {len(all_perms)}")

    chosen = all_perms[perm - 1]  # zero-based index

    # Build Vi -> dataset_var mapping
    return {
        f"V{i+1}": dataset_vars[idx]
        for i, idx in enumerate(chosen)
    }

import re

def replace_variables_with_names(program_str: str, mapping: dict) -> str:
    """
    Replace occurrences of Vi in the program with their corresponding
    dataset variable names from the mapping dict.

    mapping must be like:
        {"V1": "skill", "V2": "belief", "V3": "result"}

    The replacement is done safely via regex to match whole variables only.
    """

    # Sort keys by descending length so V12 is replaced before V1
    # (avoids partial replacements)
    keys_sorted = sorted(mapping.keys(), key=lambda k: -len(k))

    result = program_str

    for key in keys_sorted:
        # \b ensures whole-word match (V1 but not V10)
        pattern = r"\b" + re.escape(key) + r"\b"
        replacement = mapping[key]
        result = re.sub(pattern, replacement, result)

    return result


def likelihood_of_program_wrt_data(input_p, data_size = 500, program = params['PROGRAM_NAME'] ):

    perm, input_p = extract_perm_and_strip_program(input_p)
    #p = normalize_program_by_blocks(p)
    data_var_list, scm = dgp.get_vars(program)
    mapping = perm_to_mapping(perm, data_var_list)
    p = preprocess_program(input_p)
    p = replace_variables_with_names(p, mapping)
    
    #data = dgp.generate_interventional_dataset(scm, data_var_list, data_size)
    datasets_dir = path.join(path.dirname(__file__), "datasets")
    dataset_file = f"{program}.csv"
    data = np.loadtxt(path.join(datasets_dir, dataset_file), delimiter=',')


    time_start = timeit.time()
    # Calculate the likelihood of the data
    likelihood = compute_likelihood(p, data_var_list, data)
    time_end = timeit.time()
    #print(f"Time to compute likelihood: {time_end - time_start} seconds")
    if (time_end - time_start) > 20:
        print("Likelihood computation took too long:")
        print(time_end - time_start)
        print(p)

    time_start = timeit.time()
    if(params['INTERVENTIONAL_FITNESS']):
        #intervention_list = interventions.choose_interventions(data, mapping, max_interventions=params['NUM_INTERVENTIONS'])
        intervention_list = dgp.get_intervention_list(program)
        for var, value in intervention_list:
            #data_intervened = dgp.generate_interventional_dataset(scm, data_var_list, 1000, intervention={var: value})
            # Read the dataset from src/fitness/datasets, relative to this file.
            
            dataset_file = f"{program}_intervention_{var}_{value}.csv"
            data_intervened = np.loadtxt(path.join(datasets_dir, dataset_file), delimiter=',')

            #use the mapping to get the Vi corresponding to var
            vi_var = None
            for k, v in mapping.items():
                if v == var:
                    vi_var = k
                    break

            program_intervened = interventions.apply_intervention_to_program(input_p, vi_var, value)
            program_intervened = preprocess_program(program_intervened)
            program_intervened = replace_variables_with_names(program_intervened, mapping)
            interventional_likelihood = compute_likelihood(program_intervened, data_var_list, data_intervened)

            likelihood += interventional_likelihood
    time_end = timeit.time()
    #print(f"Time to compute interventional likelihoods: {time_end - time_start} seconds")
    # Calculate fitness
    fitness = likelihood 
    #if not isfinite(fitness.item()):
        
    return fitness.item(), p 


import re
from typing import List, Tuple

# ============================================================
# Regex helpers
# ============================================================

_dist_re = re.compile(r'^\s*(gm\s*\(|uniform\s*\(|bern\s*\()', re.IGNORECASE)
_number_re = re.compile(r'^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$')
_varv_re = re.compile(r'^V\d+$')
_temp_re = re.compile(r'^TEMP\d+$')

VAR_RE = re.compile(r"\bV([0-9]+)\b")

# ============================================================
# Small helpers
# ============================================================

def is_distribution(token: str) -> bool:
    return bool(_dist_re.match(token.strip()))

def is_number_token(token: str) -> bool:
    return bool(_number_re.match(token.strip()))

def _strip_unary(token: str) -> str:
    return re.sub(r'^[+-]', '', token.strip())

def is_varv(token: str) -> bool:
    return bool(_varv_re.match(_strip_unary(token)))

def is_temp(token: str) -> bool:
    return bool(_temp_re.match(_strip_unary(token)))

def is_variable(token: str) -> bool:
    return is_varv(token) or is_temp(token)


def sanitize_distribution_token(token: str) -> str:
    """
    With your new grammar, gm pos params have no leading '+', so this is mostly optional.
    Still, keep it to guarantee:
      uniform([-0.1, +7.0], 2) -> uniform([-0.1, 7.0], 2)
    """
    t = token.strip()
    if not is_distribution(t):
        return t
    return re.sub(r'(?<![eE])\+(\d|\.)', r'\1', t)

def prepare_factors_for_temp_emission(factors: List[str]) -> Tuple[List[str], bool]:
    """
    Prepare factors for TEMP chain emission with the rule:
      - If the product is negative AND it involves >=2 variable-like factors,
        do NOT place '-' on a variable. Emit unsigned product, then negate TEMP.
    Returns: (factors_for_multiplication, needs_post_negation)
    """
    overall_sign = 1
    num_prod = 1.0
    has_num = False

    unsigned_vars: List[str] = []
    other_tokens: List[str] = []
    var_count = 0

    for f in factors:
        f = f.strip()
        if not f:
            continue

        # numeric factor
        if is_number_token(f):
            v = float(f)
            if v < 0:
                overall_sign *= -1
                v = -v
            num_prod *= v
            has_num = True
            continue

        # variable-like factor (Vn or TEMPn), possibly signed
        if is_variable(f):
            var_count += 1
            if f.startswith('-'):
                overall_sign *= -1
                unsigned_vars.append(f[1:])
            elif f.startswith('+'):
                unsigned_vars.append(f[1:])
            else:
                unsigned_vars.append(f)
            continue

        # anything else (rare in your TEMP products, but keep safe)
        if f.startswith('+'):
            f = f[1:].lstrip()
        other_tokens.append(f)

    out: List[str] = []

    # fold numeric product into a single leading number if any
    if has_num:
        if overall_sign < 0:
            num_prod = -num_prod
            overall_sign = 1
        out.append(format_number(num_prod))

    out.extend(unsigned_vars)
    out.extend(other_tokens)

    # Decide where the remaining negative sign goes
    needs_post_neg = False
    if overall_sign < 0:
        if var_count >= 2:
            # NEW RULE: can't put '-' on variables inside multiplication
            needs_post_neg = True
        else:
            # For single-variable products, "-V1" is still allowed
            if out and is_variable(out[0]) and not out[0].startswith('-'):
                out[0] = "-" + out[0]
            else:
                # fallback
                out.insert(0, "-1")

    return out, needs_post_neg


def split_unary_minus_atom(expr: str) -> Tuple[bool, str]:
    """
    Detect RHS of the form '-<atom>' where <atom> is a variable/TEMP/number.
    Returns (True, atom_without_minus) if matched, else (False, expr).
    """
    s = expr.strip()
    if not s.startswith('-'):
        return False, expr
    rest = s[1:].lstrip()

    # unary-negative variable or TEMP
    if is_variable(rest):
        return True, rest

    # unary-negative number
    if is_number_token(rest):
        return True, rest

    return False, expr



# ============================================================
# Depth-aware splitters for the new grammar
# ============================================================

def split_top_level_by_spaced_ops(expr: str, ops: List[str]) -> List[Tuple[str, str]]:
    """
    Split expr by spaced operators (e.g. [" + ", " - "]) at top level only.
    Returns: [(op, chunk), ...] where op is "" for first chunk, then the operator string.
    """
    expr = expr.strip()
    if not expr:
        return []

    # sort longer ops first (e.g. " <= " before " < " if you reuse it)
    ops = sorted(ops, key=len, reverse=True)

    parts: List[Tuple[str, str]] = []
    buf = []
    depth_paren = 0
    depth_square = 0

    i = 0
    current_op = ""  # op that led to current chunk
    L = len(expr)

    while i < L:
        c = expr[i]
        if c == '(':
            depth_paren += 1
        elif c == ')':
            depth_paren -= 1
        elif c == '[':
            depth_square += 1
        elif c == ']':
            depth_square -= 1

        if depth_paren == 0 and depth_square == 0:
            matched = None
            for op in ops:
                if expr.startswith(op, i):
                    matched = op
                    break
            if matched is not None:
                chunk = "".join(buf).strip()
                parts.append((current_op, chunk))
                buf = []
                current_op = matched.strip()  # "+" or "-" or "*"
                i += len(matched)
                continue

        buf.append(c)
        i += 1

    chunk = "".join(buf).strip()
    parts.append((current_op, chunk))
    return parts

def split_sum(expr: str) -> List[Tuple[str, str]]:
    """
    Split into (op, term) where op in {"", "+", "-"}.
    Relies on grammar: binary +/− are emitted as ' + ' / ' - '.
    """
    raw = split_top_level_by_spaced_ops(expr, [" + ", " - "])
    # normalize ops
    out = []
    for op, chunk in raw:
        if not chunk:
            continue
        out.append((op if op else "", chunk))
    return out

def split_product(term: str) -> List[str]:
    """
    Split term into factors by ' * ' at top level.
    """
    raw = split_top_level_by_spaced_ops(term, [" * "])
    # raw gives (op, chunk) but op is "" for first, "*" afterwards
    factors = []
    for _, chunk in raw:
        chunk = chunk.strip()
        if chunk:
            factors.append(sanitize_distribution_token(chunk))
    return factors

# ============================================================
# Product normalization: move sign to beginning, fold numeric factors
# ============================================================

def normalize_product_signs(raw_factors: List[str]) -> List[str]:
    """
    - Pull unary signs off signed atoms (+V2, -V2, -1.23e-2)
    - Combine all numeric factors into ONE leading numeric factor
    - Move leftover negative sign to:
        * the numeric factor if present, else
        * the first variable factor (e.g. -V1 * V2), else
        * prepend -1
    """
    overall_sign = 1
    num_prod = 1.0
    has_num = False
    rest: List[str] = []

    for f in raw_factors:
        f = sanitize_distribution_token(f.strip())
        if not f:
            continue

        # signed number
        if is_number_token(f):
            v = float(f)
            if v < 0:
                overall_sign *= -1
                v = -v
            num_prod *= v
            has_num = True
            continue

        # signed var/temp
        if is_variable(f):
            if f.startswith('-'):
                overall_sign *= -1
                rest.append(f[1:])
            elif f.startswith('+'):
                rest.append(f[1:])
            else:
                rest.append(f)
            continue

        # unknown token: keep it; if it starts with '+', drop it cosmetically
        if f.startswith('+'):
            f = f[1:].lstrip()
        rest.append(f)

    out: List[str] = []

    if has_num:
        if overall_sign < 0:
            num_prod = -num_prod
            overall_sign = 1
        out.append(format_number(num_prod))

    out.extend(rest)

    if overall_sign < 0:
        if out and is_variable(out[0]) and not out[0].startswith('-'):
            out[0] = "-" + out[0]  # allow -V1 as first factor
        else:
            out.insert(0, "-1")

    return out

def order_pair_for_mul(a: str, b: str) -> Tuple[str, str]:
    """Put numeric first if exactly one operand is numeric."""
    a_s, b_s = a.strip(), b.strip()
    a_is = is_number_token(a_s)
    b_is = is_number_token(b_s)
    if a_is and not b_is:
        return a_s, b_s
    if b_is and not a_is:
        return b_s, a_s
    return a_s, b_s

def format_product(factors: List[str]) -> str:
    """
    Canonical product string with '*' separators, after sign normalization.
    """
    facs = normalize_product_signs(factors)
    return " * ".join(facs) if facs else "0"

# ============================================================
# Add/Sub emission normalization (fixes: X = X - -0.01)
# ============================================================

def emit_addsub(indent: str, lhs: str, op: str, expr: str) -> str:
    """
    Emit: lhs = lhs <op> expr;
    Simplify:
      - lhs - -a -> lhs + a
      - lhs + -a -> lhs - a
      - strip unary '+' on expr
    """
    op = op.strip()
    e = expr.strip()

    if e.startswith('+'):
        e = e[1:].lstrip()

    if e.startswith('-'):
        e2 = e[1:].lstrip()
        if op == '+':
            return f"{indent}{lhs} = {lhs} - {e2};"
        if op == '-':
            return f"{indent}{lhs} = {lhs} + {e2};"

    return f"{indent}{lhs} = {lhs} {op} {e};"

# ============================================================
# Main preprocessor (ONLY V variables)
# ============================================================

def pre_process_instructions(program: str) -> str:
    """
    Robust for your new grammar:
      - sums use spaced '+/-'
      - products use spaced '*'
      - signed atoms have no spaces (e.g., -V2, +9.64e+2)
    """
    lines = program.split('\n')
    out: List[str] = []
    temp_counter = 0

    def new_temp() -> str:
        nonlocal temp_counter
        t = f"TEMP{temp_counter}"
        temp_counter += 1
        return t

    for line in lines:
        raw_line = line.rstrip('\n')
        if not raw_line.strip():
            continue

        indent_len = len(raw_line) - len(raw_line.lstrip())
        indent = raw_line[:indent_len]
        content = raw_line[indent_len:].rstrip()

        # split by ';' but keep structure
        segs = [s.strip() for s in content.split(';') if s.strip()]
        for seg in segs:
            if not re.match(r'^V\d+\s*=', seg):
                s = seg.strip()

                # Do NOT add ';' after block openers/closers
                if s.endswith('{') or s in ('}', 'else {') or s.endswith('} else {'):
                    out.append(f"{indent}{s}")
                # Keep ';' for end-if (your syntax uses it)
                elif s.lower() == 'end if':
                    out.append(f"{indent}{s};")
                else:
                    # default: keep ';' for other statements
                    out.append(f"{indent}{s};")

                continue


            left, rhs = seg.split("=", 1)
            left = left.strip()
            rhs = rhs.strip()

            # 1) split sum into terms by spaced +/-
            terms = split_sum(rhs)

            # 2) create TEMP map for products with any V-factor and len>1
            temp_map: dict[Tuple[str, ...], str] = {}
            pre_temps: List[Tuple[str, Tuple[str, ...]]] = []

            for op, term in terms:
                # op is "" for first, "+" or "-"
                factors = split_product(term)
                factors = normalize_product_signs(factors)
                key = tuple(factors)

                if len(factors) > 1 and any(is_varv(f) for f in factors):
                    if key not in temp_map:
                        t = new_temp()
                        temp_map[key] = t
                        pre_temps.append((t, key))

            # 3) emit TEMP chains
            # emit TEMP chains (explicit '*', numeric-first at every step)
            for tname, key in pre_temps:
                raw_factors = list(key)

                # NEW: enforce "no signed vars in var*var products"
                factors, post_negate = prepare_factors_for_temp_emission(raw_factors)

                # build multiplication chain
                a, b = order_pair_for_mul(factors[0], factors[1])
                out.append(f"{indent}{tname} = {a} * {b};")

                for f in factors[2:]:
                    a, b = order_pair_for_mul(tname, f)
                    out.append(f"{indent}{tname} = {a} * {b};")

                # NEW: apply sign after the product if needed
                if post_negate:
                    out.append(f"{indent}{tname} = 0 - {tname};")


            # 4) emit final assignment as sequential updates
            first = True
            for op, term in terms:
                factors = normalize_product_signs(split_product(term))
                key = tuple(factors)

                expr = temp_map.get(key, format_product(factors))

                if first:
                    # skip no-op initialization: V = V
                    if expr == left and (op == "" or op == "+"):
                        first = False
                        continue

                    # NEW RULE: catch unary negatives that come with no spaced '-' operator, e.g. "T=-F" or "T=-3.2"
                    # (in this case split_sum() returns op == "" and expr == "-F")
                    is_unary_neg, atom = split_unary_minus_atom(expr)
                    if is_unary_neg and (op == "" or op == "+"):
                        out.append(f"{indent}{left} = 0;")
                        out.append(emit_addsub(indent, left, "-", atom))
                        first = False
                        continue

                    if op == "-":
                        # IMPORTANT: distributions cannot be unary-negated
                        out.append(f"{indent}{left} = 0;")
                        out.append(emit_addsub(indent, left, "-", expr))  # becomes: left = left - dist
                        first = False
                        continue

                    # first positive term
                    out.append(f"{indent}{left} = {expr};")
                    first = False
                    continue

                # subsequent terms always use add/sub updates
                out.append(emit_addsub(indent, left, op if op in {"+", "-"} else "+", expr))

    return "".join(out)

# ============================================================
# Optional: number conversion (keeps your behavior)
# ============================================================
from decimal import Decimal, InvalidOperation, getcontext

def _decimal_str_fixed(num_str: str, ndigits: int = 4) -> str:
    """
    Convert a numeric literal to fixed-point decimal with rounding.
    - No scientific notation
    - Rounded to `ndigits`
    - Trailing zeros trimmed
    """
    s = num_str.strip()
    getcontext().prec = 50  # enough precision for safe rounding

    try:
        d = Decimal(s)
    except InvalidOperation:
        return num_str

    # Round to fixed decimal places
    q = Decimal("1").scaleb(-ndigits)   # e.g. 1e-4
    d = d.quantize(q)

    # Fixed-point formatting
    out = format(d, "f")

    # Trim trailing zeros and trailing dot
    if "." in out:
        out = out.rstrip("0").rstrip(".")

    # Avoid "-0"
    if out == "-0":
        out = "0"

    return out


def convert_numbers_to_floats(program: str) -> str:
    """
    Convert all numeric literals to fixed-point decimals rounded to 4 places.
    - No scientific notation
    - Doesn't touch variable names (V1, TEMP0, etc.)
    """
    number_pattern = re.compile(
        r'(?<![A-Za-z0-9_])([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)(?![A-Za-z0-9_])'
    )

    def repl(m):
        return _decimal_str_fixed(m.group(0), ndigits=4)

    return number_pattern.sub(repl, program)

getcontext().prec = 50

def format_number(x) -> str:
    """
    Canonical numeric formatter:
    - accepts float, Decimal, or str
    - fixed-point
    - rounded to 4 decimals
    - never scientific notation
    """
    d = Decimal(str(x))
    q = Decimal("1e-4")   # 4 decimal places
    d = d.quantize(q)

    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s
