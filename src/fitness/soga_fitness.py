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

def compute_likelihood(output_dist, data_var_list, data):
    """ computes the likelihood of output_dist with respect to variables data_var_list sampled in data """

    data = torch.tensor(data)
    likelihood = 0
    # extract indexes of the variables in the data
    try:
        data_var_index = [output_dist.var_list.index(element) for element in data_var_list ]
    except ValueError:  # if the program doesn't have all the variables we are using for the likelihood
            return torch.tensor(-np.inf)
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
    
    return torch.sum(torch.log(likelihood))/len(data)

class soga_fitness(base_ff):
    """Fitness function for finding the length of the shortest path between
    two nodes in a grade compared to the known shortest path. Takes a path (
    list of node labels as strings) and returns fitness. Penalises output
    that is not the same length as the target."""

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()
        

    def evaluate(self, ind, **kwargs):
        self.default_fitness = -np.inf
        p = ind.phenotype
        #p = smooth_program(p)
        #print("\n" + p)
        #print("\n -----------------------------------------")

        fitness = 0
        #timer = threading.Timer(10, timeout_handler)
        try:
            #timer.start()
            fitness = likelihood_of_program_wrt_data(p)
        except TimeoutException as e:
            print("Caught TimeoutException")
            fitness = self.default_fitness
        except:
            fitness = self.default_fitness
            #I do not define the indiviaduals as invalid in order to allow crossover
            #if not hasattr(params['FITNESS_FUNCTION'], "multi_objective"):
                #stats['invalids'] += 1
        #finally:
            #timer.cancel()
        
    
        return fitness

def generate_list():
    return [random.randint(0, round(random.random() * 90 + 10)) for i in range(9)]

def preprocess_program(program):
    p = pre_process_instructions(program)
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
        normalized_pi_list = [pi / pi_sum for pi in pi_list] if pi_sum != 0 else pi_list
        
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

def smooth_program(program_text):
    """
    Finds and modifies assignment lines where the variable is a letter or alphanumeric
    and the right-hand side does not contain 'gm' or 'uniform'.
    
    Args:
        program_text (str): The text of the program to analyze and modify.
    
    Returns:
        str: The modified program text.
    """
    # Define the regex pattern to identify assignments
    pattern = r'^(\s*([a-zA-Z]\w*)\s*=\s*([^;]*))(;?)'
    
    # Split the program into lines
    lines = program_text.splitlines()
    modified_lines = []
    
    for line in lines:
        match = re.match(pattern, line)
        if match:
            full_assignment, var, expression, semicolon = match.groups()
            # Check if 'gm' or 'uniform' is in the expression
            if 'gm' not in expression and 'uniform' not in expression:
                # Modify the assignment
                modified_assignment = f"{full_assignment} + gauss(0., 0.05){semicolon}"
                modified_lines.append(modified_assignment)
            else:
                # Keep the original line
                modified_lines.append(line)
        else:
            # Keep non-matching lines unchanged
            modified_lines.append(line)
    
    # Join the lines back into a single text
    return '\n'.join(modified_lines)   


def likelihood_of_program_wrt_data(p, data_size = 500, program = params['PROGRAM_NAME'] ):
    
    p = preprocess_assign_conditionals(p)

    if not check_no_assignment_after_use(p):
        #stats['invalids'] += 1
        return -np.inf
    if not check_no_reassignment_of_U(p):
        #stats['invalids'] += 1
        return -np.inf
    
    if not check_all_rhs_assigned_before_use(p):
        #stats['invalids'] += 1
        return -np.inf
    
    if not check_boolean_vars_previously_assigned(p):
        #stats['invalids'] += 1
        return -np.inf

     # Preprocess program
    p = preprocess_program(p)
    data_var_list, dependencies, weights = dgp.get_vars(program)
    dependencies_benefit = 0
    data = dgp.generate_dataset(program, data_size)
    
    # Computes output distribution of the program
    compiledText=compile2SOGA_text(p)
    cfg = produce_cfg_text(compiledText)
    try:                                
        output_dist = start_SOGA(cfg)
    except IndexError: # program has no valid paths
        #stats['invalids'] += 1
        return -np.inf

    # Calculate the benefit of dependencies
    if(params['DEPENDENCIES_BENEFIT']):
        for key, values in dependencies.items():
            key_index = output_dist.var_list.index(key)
            for value in values:
                value_index = output_dist.var_list.index(value)
                cov_value = output_dist.gm.cov()[key_index, value_index]
                #if((output_dist.gm.cov()[key_index, key_index]!= 0) & (output_dist.gm.cov()[value_index, value_index]!= 0) ):
                    #cov_value = cov_value/(torch.sqrt(np.abs(output_dist.gm.cov()[key_index, key_index] * output_dist.gm.cov()[value_index, value_index])))
                #if cov_value < 1e-10:
                    #raise ValueError(f"Variable {key} and {value} have covariance 0")
                dependencies_benefit += weights[key] * np.log(np.abs(cov_value))

    # Calculate the likelihood of the data
    likelihood = compute_likelihood(output_dist, data_var_list, data)

    # Calculate fitness
    fitness = likelihood + dependencies_benefit
    if not isfinite(fitness.item()):
        if check_all_data_vars_assigned(p, data_var_list):
            return -1e10
    return fitness.item()

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


# match assignment of form "   V3 = ..." or "TEMP2=..."
assign_re = re.compile(r'^\s*(?P<lhs>(?:V|U|TEMP)\d+)\s*=')

# match variable names appearing on the RHS
var_re = re.compile(r'\b(V\d+|U\d+|TEMP\d+)\b')


def extract_lhs_rhs(instr: str):
    """
    Extract (lhs, rhs_vars) from a single instruction IF it is an assignment.
    Otherwise returns (None, empty_set).
    """
    m = assign_re.match(instr)
    if not m:
        return None, set()

    lhs = m.group("lhs")

    # RHS = everything after '='
    rhs = instr.split("=", 1)[1]

    # extract variables appearing on the RHS
    rhs_vars = set(var_re.findall(rhs))

    # self-reference (V3 = V3 + 1) is NOT a violation
    rhs_vars.discard(lhs)

    return lhs, rhs_vars


def check_no_assignment_after_use(program: str):
    """
    Checks the rule: no variable may be assigned AFTER it has already been used
    in any previous RHS.

    Returns:
        True if valid
        False if invalid
    """
    # split instructions on semicolon, but KEEP the original text intact
    instructions = [part for part in program.split(";") if part.strip()]

    used_vars = set()

    for instr in instructions:
        instr = instr.rstrip()  # remove only trailing spaces (keep indentation)

        lhs, rhs_vars = extract_lhs_rhs(instr)

        # add RHS variables to the used set
        used_vars.update(rhs_vars)

        if lhs is not None:
            # rule violation: lhs was previously used
            if lhs in used_vars:
                return False

            # note: we don't add lhs to used_vars here,
            # because only RHS appearances count as "use"

    return True


import re

assign_re = re.compile(r'^\s*(?P<lhs>(?:V|U|TEMP)\d+)\s*=')
rhs_var_re = re.compile(r'\b(V\d+|U\d+|TEMP\d+)\b')


def extract_lhs_rhs(instr: str):
    """Extract LHS variable and RHS variable set from an assignment line."""
    m = assign_re.match(instr)
    if not m:
        return None, set()

    lhs = m.group("lhs")
    rhs = instr.split("=", 1)[1]
    rhs_vars = set(rhs_var_re.findall(rhs))
    rhs_vars.discard(lhs)  # remove self reference
    return lhs, rhs_vars


def check_no_reassignment_of_U(program: str):
    """
    Rule:
      (1) Each Ui must be assigned at most once.
      (2) Each Ui may appear in RHS of at most one *V-assignment*.
    """
    instructions = [p for p in program.split(";") if p.strip()]

    assigned_U = set()
    used_by_V = {}   # map U -> name of V that used it

    for instr in instructions:
        instr = instr.rstrip()

        lhs, rhs_vars = extract_lhs_rhs(instr)
        if lhs is None:
            continue

        # ---------------------------------------
        # (1) Ui assigned only once
        # ---------------------------------------
        if lhs.startswith("U"):
            if lhs in assigned_U:
                return False
            assigned_U.add(lhs)

        # ---------------------------------------
        # (2) Ui used in at most one V-assignment
        # ---------------------------------------
        if lhs.startswith("V"):
            for var in rhs_vars:
                if var.startswith("U"):
                    if var not in used_by_V:
                        used_by_V[var] = lhs
                    else:
                        # Already used by a different V
                        if used_by_V[var] != lhs:
                            return False

    return True

assign_re = re.compile(r'^\s*(?P<lhs>(?:V|U|TEMP)\d+)\s*=')
var_re = re.compile(r'\b(V\d+|U\d+|TEMP\d+)\b')


def extract_lhs_rhs(instr: str):
    """Extract (lhs, rhs_vars) from assignment. Otherwise (None, empty)."""
    m = assign_re.match(instr)
    if not m:
        return None, set()

    lhs = m.group("lhs")
    rhs = instr.split("=", 1)[1]
    rhs_vars = set(var_re.findall(rhs))

    # self-reference is allowed (V3 = V3 + 1)
    rhs_vars.discard(lhs)

    return lhs, rhs_vars


def check_all_rhs_assigned_before_use(program: str):
    """
    Ensures that every variable on the RHS has been assigned
    in a previous instruction.

    Returns:
        True  if valid
        False if any RHS variable is used before assignment
    """
    instructions = [part for part in program.split(";") if part.strip()]
    assigned = set()
    line_number = 0

    for instr in instructions:
        line_number += 1
        instr = instr.rstrip()

        lhs, rhs_vars = extract_lhs_rhs(instr)

        # Check all RHS vars must be already assigned
        for var in rhs_vars:
            if var not in assigned:
                return False  # early violation

        # If assignment, record LHS as now assigned
        if lhs is not None:
            assigned.add(lhs)

    return True


# Regex for:  ASSIGN[V7, U3]
assign_directive_re = re.compile(
    r'ASSIGN\s*\[\s*(V\d+)\s*,\s*(U\d+)\s*\]',
    re.IGNORECASE
)

# Regex to extract expressions inside branch braces { : ... : }
branch_re = re.compile(
    r'\{\s*:\s*(.*?)\s*:\s*\}',
    re.DOTALL
)

def preprocess_assign_conditionals(program: str) -> str:
    """
    STEP 0: Expand ASSIGN[Vx, Uy] conditionals into explicit assignments:
       if cond ASSIGN[Vx, Uy] { : EXPR1 : } else { : EXPR2 : } end if;
    becomes:
       if cond { Vx = EXPR1 + Uy; } else { Vx = EXPR2 + Uy; } end if;
    """

    # Find every conditional with ASSIGN[…]
    def repl(match):
        full = match.group(0)

        # Extract variable names
        assign_match = assign_directive_re.search(full)
        if not assign_match:
            return full  # shouldn't happen

        v_var = assign_match.group(1)
        u_var = assign_match.group(2)

        # Extract both branch bodies
        branches = branch_re.findall(full)
        if len(branches) != 2:
            # malformed conditional; return unchanged
            return full

        expr1, expr2 = branches

        # Clean expressions
        expr1 = expr1.strip().rstrip(";")
        expr2 = expr2.strip().rstrip(";")

        # Build rewritten branches
        new_branch1 = f"{{ {v_var} = {expr1} + {u_var}; }}"
        new_branch2 = f"{{ {v_var} = {expr2} + {u_var}; }}"

        # Rebuild conditional WITHOUT the ASSIGN[...] directive
        rewritten = assign_directive_re.sub("", full)

        # Replace original branch bodies with new ones
        rewritten = branch_re.sub(lambda m, cnt=[0]:
                                  new_branch1 if cnt.append(1) or len(cnt)==2 else new_branch2,
                                  rewritten, count=2)

        # Final cleanup of extra spaces
        rewritten = re.sub(r'\s+', ' ', rewritten)

        return rewritten

    # Pattern to detect whole conditional blocks containing ASSIGN[…]
    conditional_pattern = re.compile(
        r'if\s+[^{}]+ASSIGN\s*\[[^\]]+\]\s*\{[^{}]*\}\s*else\s*\{[^{}]*\}\s*end if;',
        re.IGNORECASE | re.DOTALL
    )

    # Apply transformation
    return conditional_pattern.sub(repl, program)

# Matches: if <condition> {
if_re = re.compile(
    r'\bif\s+(?P<cond>.*?)\s*\{',
    re.IGNORECASE | re.DOTALL
)

# Variables V#, U#, TEMP#
var_re = re.compile(r'\b(V\d+|U\d+|TEMP\d+)\b')

# Assignment on LHS
assign_re = re.compile(r'^\s*(?P<lhs>(?:V|U|TEMP)\d+)\s*=')


def extract_assigned_var(instr: str):
    """Return LHS variable name or None."""
    m = assign_re.match(instr)
    return m.group("lhs") if m else None


def check_boolean_vars_previously_assigned(program: str):
    """
    Ensures all variables used inside boolean conditions
    have been assigned before their use.

    Returns:
        True  if valid
        False if any variable in a condition was used before assignment
    """
    # Split program into instructions (semicolon separated)
    instructions = [p for p in program.split(";") if p.strip()]

    assigned = set()
    line_no = 0

    for instr in instructions:
        line_no += 1
        instr_clean = instr.strip()

        # Check if this line contains an "if <cond> {"
        m = if_re.search(instr_clean)
        if m:
            cond = m.group("cond")
            vars_in_cond = set(var_re.findall(cond))

            # Check all vars have been previously assigned
            for v in vars_in_cond:
                if v not in assigned:
                    # violation
                    return False

        # Handle assignment updating assigned set
        lhs = extract_assigned_var(instr_clean)
        if lhs:
            assigned.add(lhs)

    return True


assign_re = re.compile(r'^\s*(?P<lhs>(?:V|U|TEMP)\d+)\s*=')

def extract_lhs(instr: str):
    """Returns the assigned variable on the left side of '=' or None."""
    m = assign_re.match(instr)
    return m.group("lhs") if m else None


def check_all_data_vars_assigned(program: str, data_var_list):
    """
    Checks whether every variable in data_var_list appears at least once
    on the left-hand side of an assignment in the program.

    Returns:
        True  if all variables are assigned somewhere
        False otherwise
    """
    # split on semicolons, do NOT strip indentation
    instructions = [p for p in program.split(";") if p.strip()]

    assigned = set()

    for instr in instructions:
        lhs = extract_lhs(instr)
        if lhs:
            assigned.add(lhs)

    # all variables must be in 'assigned'
    return all(var in assigned for var in data_var_list)
