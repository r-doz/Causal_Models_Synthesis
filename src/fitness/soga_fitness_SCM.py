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

    
    # Computes output distribution of the program
    compiledText=compile2SOGA_text(p)
    cfg = produce_cfg_text(compiledText)
    try:                                
        output_dist = start_SOGA(cfg)
    except IndexError: # program has no valid paths
        #stats['invalids'] += 1
        print("Program has no valid paths")
        return torch.tensor(-1e6)
    
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
        new_uniform = f'uniform([{a:.6f}, {new_b:.6f}], {c})'
        
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
    
    data = dgp.generate_interventional_dataset(scm, data_var_list, data_size)


    # Calculate the likelihood of the data
    likelihood = compute_likelihood(p, data_var_list, data)

    if(params['INTERVENTIONAL_FITNESS']):
        #intervention_list = interventions.choose_interventions(data, mapping, max_interventions=params['NUM_INTERVENTIONS'])
        intervention_list = dgp.get_intervention_list(program)
        for var, value in intervention_list:
            #data_intervened = dgp.generate_interventional_dataset(scm, data_var_list, 1000, intervention={var: value})
            # Read the dataset from src/fitness/datasets, relative to this file.
            datasets_dir = path.join(path.dirname(__file__), "datasets")
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

    # Calculate fitness
    fitness = likelihood 
    #if not isfinite(fitness.item()):
        
    return fitness.item(), p 

import re
from typing import List, Tuple

# -------------------------
# Regex helpers
# -------------------------
_dist_re   = re.compile(r'^\s*(gm\s*\(|uniform\s*\(|bern\s*\()', re.IGNORECASE)
_number_re = re.compile(r'^[+-]?\d+(\.\d+)?$')   # normalized numbers only (no surrounding spaces)
_varv_re    = re.compile(r'^V\d+$')
_varu_re    = re.compile(r'^U\d+$')
_temp_re    = re.compile(r'^TEMP\d+$')

# -------------------------
# Token predicates
# -------------------------
def is_distribution(token: str) -> bool:
    return bool(_dist_re.match(token.strip()))

def is_number_token(token: str) -> bool:
    return bool(_number_re.match(token.strip()))

def is_varv(token: str) -> bool:
    return bool(_varv_re.match(token.strip()))

def is_varu(token: str) -> bool:
    return bool(_varu_re.match(token.strip()))

def is_temp(token: str) -> bool:
    return bool(_temp_re.match(token.strip()))

def is_variable(token: str) -> bool:
    return is_varv(token) or is_varu(token) or is_temp(token)

# -------------------------
# Parsing helpers
# -------------------------
def split_factors(product_str: str) -> List[str]:
    """Split top-level product factors (respecting parentheses and square brackets)."""
    parts = []
    buf = ""
    depth_square = 0
    depth_paren = 0
    for c in product_str:
        if c == '[': depth_square += 1
        elif c == ']': depth_square -= 1
        elif c == '(': depth_paren += 1
        elif c == ')': depth_paren -= 1

        if c == '*' and depth_square == 0 and depth_paren == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += c
    if buf.strip():
        parts.append(buf.strip())
    return parts

def split_top_level_plus_minus(expr: str) -> List[Tuple[str,str]]:
    """
    Split expr into signed top-level terms.
    Return list of tuples (sign, term) where sign is '+' or '-'.
    """
    terms = []
    buf = ""
    depth_square = 0
    depth_paren = 0
    for c in expr:
        if c == '[': depth_square += 1
        elif c == ']': depth_square -= 1
        elif c == '(': depth_paren += 1
        elif c == ')': depth_paren -= 1

        if (c == '+' or c == '-') and depth_square == 0 and depth_paren == 0 and buf:
            terms.append(buf.strip())
            buf = c
        else:
            buf += c
    if buf.strip():
        terms.append(buf.strip())

    signed = []
    for t in terms:
        if t.startswith('+'):
            signed.append(('+', t[1:].strip()))
        elif t.startswith('-'):
            signed.append(('-', t[1:].strip()))
        else:
            signed.append(('+', t))
    return signed

# -------------------------
# Pair canonicalization helper
# -------------------------
def order_pair_for_mul(a: str, b: str) -> Tuple[str,str]:
    """
    Canonical order for a single multiplication pair: return (lhs, rhs).
    If exactly one operand is numeric, return (number, other).
    Otherwise return (a.strip(), b.strip()) unchanged.
    """
    a_s = a.strip()
    b_s = b.strip()
    a_is_num = is_number_token(a_s)
    b_is_num = is_number_token(b_s)

    if a_is_num and not b_is_num:
        return a_s, b_s
    if b_is_num and not a_is_num:
        return b_s, a_s
    return a_s, b_s

# -------------------------
# Normalization helpers
# -------------------------
def fold_numeric_factors(factors: List[str]):
    """If all factors are numeric, fold product and return string; else return None."""
    if all(is_number_token(f.strip()) for f in factors):
        prod = 1.0
        for f in factors:
            prod *= float(f)
        return str(int(prod)) if prod.is_integer() else repr(prod)
    return None

def reorder_number_first(factors: List[str]) -> List[str]:
    """
    If exactly one non-numeric factor and >=1 numeric factors, fold numbers and put number first.
    """
    stripped = [f.strip() for f in factors]
    numbers = [f for f in stripped if is_number_token(f)]
    non_numbers = [f for f in stripped if not is_number_token(f)]

    if len(non_numbers) == 1 and len(numbers) >= 1:
        num_val = 1.0
        for n in numbers:
            num_val *= float(n)
        num_str = str(int(num_val)) if num_val.is_integer() else repr(num_val)
        return [num_str, non_numbers[0]]
    return stripped

def normalize_factors_for_key(factors: List[str]) -> Tuple[str,...]:
    """Normalized, hashable tuple used as key (spacing + numbers-first)."""
    facs = [f.strip() for f in factors]
    facs = reorder_number_first(facs)
    return tuple(facs)

def format_product(factors: List[str]) -> str:
    """
    Canonical string for a product of factors, used when we emit a non-TEMP product.
    """
    facs = [f.strip() for f in factors]
    folded = fold_numeric_factors(facs)
    if folded is not None:
        return folded

    if len([f for f in facs if not is_number_token(f)]) == 1:
        facs2 = reorder_number_first(facs)
        return " * ".join(facs2)

    out = [facs[0]]
    for nxt in facs[1:]:
        prev = out[-1]
        if is_number_token(prev) and not is_number_token(nxt):
            out.append(nxt)
        elif is_number_token(nxt) and not is_number_token(prev):
            out[-1] = nxt
            out.append(prev)
        else:
            out.append(nxt)
    return " * ".join(out)

# -------------------------
# Main preprocessor
# -------------------------
def pre_process_instructions(program: str) -> str:
    """
    Preprocess program.
    - Splits on newlines and semicolons, but preserves indentation.
    - Returns a single-line string, no extra ';' after '{' or 'else {'.
    """
    lines = program.split('\n')
    out_instrs: List[str] = []
    temp_counter = 0

    def new_temp():
        nonlocal temp_counter
        name = f"TEMP{temp_counter}"
        temp_counter += 1
        return name

    for line in lines:
        # Keep original line (for indentation and to know if it had trailing ';')
        raw_line = line.rstrip('\n')
        if not raw_line.strip():
            continue

        # Compute indentation and content
        indent_len = len(raw_line) - len(raw_line.lstrip())
        indent = raw_line[:indent_len]
        content = raw_line[indent_len:]

        # Detect if the line ended with a semicolon
        content_rstrip = content.rstrip()
        ends_with_semicolon = content_rstrip.endswith(';')
        # Strip the trailing semicolon (if any) before splitting segments
        if ends_with_semicolon:
            content_core = content_rstrip[:-1]
        else:
            content_core = content_rstrip

        # Split this line into segments separated by ';'
        segments = content_core.split(';')

        for i, seg in enumerate(segments):
            seg_stripped = seg.strip()
            if not seg_stripped:
                continue

            # This segment had a semicolon in the original line if:
            # - it's not the last segment of the line, OR
            # - it is last but the original line ended with ';'
            seg_had_semicolon = (i < len(segments) - 1) or (ends_with_semicolon and i == len(segments) - 1)

            instr_stripped = seg_stripped

            # Non-assignment: keep exactly as-is, re-adding semicolon only if it was in original.
            if not re.match(r'^(U|V)\d+\s*=', instr_stripped):
                if seg_had_semicolon:
                    out_instrs.append(f"{indent}{instr_stripped};")
                else:
                    out_instrs.append(f"{indent}{instr_stripped}")
                continue

            # Now we know we have an assignment U* or V*
            left, right = instr_stripped.split("=", 1)
            left = left.strip()
            rhs  = right.strip()

            # ---------- Endogenous (V*) ----------
            if left.startswith("V"):
                signed_terms = split_top_level_plus_minus(rhs)

                temp_map = {}
                pre_temps = []

                # identify all products with V* that need TEMPs
                for sign, term in signed_terms:
                    factors = split_factors(term)
                    key = normalize_factors_for_key(factors)
                    if len(factors) > 1 and any(is_varv(f) for f in factors):
                        if key not in temp_map:
                            t = new_temp()
                            temp_map[key] = t
                            pre_temps.append((t, key))

                # generate TEMP assignments
                for tname, key in pre_temps:
                    factors = list(key)
                    a, b = order_pair_for_mul(factors[0], factors[1])
                    out_instrs.append(f"{indent}{tname} = {a} * {b};")
                    for f in factors[2:]:
                        a, b = order_pair_for_mul(tname, f)
                        out_instrs.append(f"{indent}{tname} = {a} * {b};")

                # build final V assignment sequence
                first_term = True
                for sign, term in signed_terms:
                    factors = split_factors(term)
                    key = normalize_factors_for_key(factors)
                    if key in temp_map:
                        expr = temp_map[key]
                    else:
                        expr = format_product(factors)

                    if first_term:
                        if sign == '+':
                            out_instrs.append(f"{indent}{left} = {expr};")
                        else:
                            out_instrs.append(f"{indent}{left} = 0;")
                            out_instrs.append(f"{indent}{left} = {left} - {expr};")
                        first_term = False
                    else:
                        if sign == '+':
                            out_instrs.append(f"{indent}{left} = {left} + {expr};")
                        else:
                            out_instrs.append(f"{indent}{left} = {left} - {expr};")

                continue  # next segment

            # ---------- Exogenous (U*) ----------
            signed_terms = split_top_level_plus_minus(rhs)
            term_outputs: List[Tuple[str,str]] = []

            for sign, term in signed_terms:
                factors = split_factors(term)
                factors = reorder_number_first(factors)

                folded = fold_numeric_factors(factors)
                if folded is not None:
                    term_outputs.append((sign, folded))
                    continue

                if len(factors) == 1:
                    term_outputs.append((sign, factors[0]))
                    continue

                # Build chained TEMPs with pair canonicalization
                f0, f1, *rest = factors
                t0 = new_temp()
                a, b = order_pair_for_mul(f0, f1)
                out_instrs.append(f"{indent}{t0} = {a} * {b};")
                prev = t0
                for f in rest:
                    tnext = new_temp()
                    a, b = order_pair_for_mul(prev, f)
                    out_instrs.append(f"{indent}{tnext} = {a} * {b};")
                    prev = tnext
                term_outputs.append((sign, prev))

            # combine signed terms for U*
            expr_parts = []
            first_term = True
            for sign, term_expr in term_outputs:
                if first_term:
                    if sign == '-':
                        expr_parts.append(f"0 - {term_expr}")
                    else:
                        expr_parts.append(term_expr)
                    first_term = False
                else:
                    if sign == '-':
                        expr_parts.append(f"- {term_expr}")
                    else:
                        expr_parts.append(f"+ {term_expr}")

            final_rhs = " ".join(expr_parts) if expr_parts else "0"
            out_instrs.append(f"{indent}{left} = {final_rhs};")

    # Concatenate everything exactly as generated (no extra separators).
    return "".join(out_instrs)


import re

# Matches any variable Vn
VAR_RE = re.compile(r"\bV([0-9]+)\b")


def normalize_block(block: str, i: int) -> str:
    """
    Normalize all occurrences of Vn inside a <Vi> block (assignments, booleans,
    nested IFs, etc.), using the rule:
        Vn -> V[((n-1) mod (i-1)) + 1]
    for i > 1, and all Vn -> V1 when i == 1.

    IMPORTANT:
      - Does NOT change LHS "Vj" when it is immediately followed by a single '='
        (possibly with spaces). So "V2 = ..." stays "V2 = ...".
      - But "V2 == 3" is NOT treated as LHS, so V2 is normalized there.
    """

    parts = []
    last_idx = 0
    L = len(block)

    for m in VAR_RE.finditer(block):
        start, end = m.span()
        n_str = m.group(1)
        n = int(n_str)

        # Look ahead to see if this occurrence is followed by '=' (potential LHS)
        j = end
        while j < L and block[j].isspace():
            j += 1

        is_lhs = False
        if j < L and block[j] == '=':
            # Check if this is a single '=' (assignment) and NOT '==' (comparison)
            if not (j + 1 < L and block[j + 1] == '='):
                is_lhs = True

        # Copy text before the match
        parts.append(block[last_idx:start])

        if is_lhs:
            # Keep LHS variable as-is (e.g., "V2" in "V2 = ...")
            parts.append(m.group(0))
        else:
            # Normalize RHS/boolean occurrences
            if i == 1:
                new_var = "V1"
            else:
                m_idx = ((n - 1) % (i - 1)) + 1
                new_var = f"V{m_idx}"
            parts.append(new_var)

        last_idx = end

    # Remainder
    parts.append(block[last_idx:])
    return "".join(parts)

def split_into_vi_blocks(program_wo_perm: str):
    """
    Split the program into <V1>, <V2>, ... blocks.
    Each block may span multiple lines (especially IF blocks).
    """

    lines = [ln for ln in program_wo_perm.split("\n") if ln.strip()]
    blocks = []
    current = []

    nesting = 0
    for line in lines:
        stripped = line.strip()

        # Start of IF?
        if stripped.startswith("if "):
            nesting += 1
            current.append(line)
            continue

        # End of IF?
        if stripped.startswith("} end if"):
            current.append(line)
            nesting -= 1

            # Finished a whole IF block
            if nesting == 0:
                blocks.append("\n".join(current))
                current = []
            continue

        # Inside nested IF
        if nesting > 0:
            current.append(line)
            continue

        # Outside IF: line must be "Vi = expr;"
        if stripped.startswith("V") and "=" in stripped:
            # This is a full single-line Vi assignment
            if current:
                raise RuntimeError("Dangling block before assignment")
            blocks.append(line)
            continue

        raise RuntimeError(f"Unexpected line: {line}")

    if nesting != 0:
        raise RuntimeError("Unbalanced IF / end if")

    return blocks



def normalize_program_by_blocks(program_wo_perm: str) -> str:

    blocks = split_into_vi_blocks(program_wo_perm)
    normalized = []

    for i, block in enumerate(blocks, start=1):
        normalized.append(normalize_block(block, i))

    return "\n".join(normalized)
