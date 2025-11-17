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
    return fitness.item()


# Regex helpers
_dist_re = re.compile(r'^\s*(gm\s*\(|uniform\s*\(|bern\s*\()', re.IGNORECASE)
_number_re = re.compile(r'^\s*[-+]?\d+(\.\d+)?\s*$')
_varv_re = re.compile(r'^V\d+$')
_varu_re = re.compile(r'^U\d+$')
_temp_re = re.compile(r'^TEMP\d+$')

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

def split_factors(product_str: str):
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
    if buf.strip(): parts.append(buf.strip())
    return parts

def join_factors(factors):
    return " * ".join(factors)

def reorder_number_first(factors):
    numbers = [f for f in factors if is_number_token(f)]
    non_numbers = [f for f in factors if not is_number_token(f)]
    if len(non_numbers) == 1 and len(numbers) >= 1:
        # fold numbers first
        num = 1.0
        for n in numbers:
            num *= float(n)
        if abs(num - round(num)) < 1e-12:
            num_str = str(int(round(num)))
        else:
            num_str = repr(num)
        return [num_str] + non_numbers
    return factors

def split_top_level_plus(expr: str):
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
    if buf.strip(): terms.append(buf.strip())
    signed_terms = []
    for t in terms:
        if t.startswith('+'): signed_terms.append(('+', t[1:].strip()))
        elif t.startswith('-'): signed_terms.append(('-', t[1:].strip()))
        else: signed_terms.append(('+', t))
    return signed_terms

def pre_process_instructions(program: str) -> str:
    parts = [p.strip() for p in program.split(';') if p.strip()]
    out_instrs = []
    temp_counter = 0

    def new_temp():
        nonlocal temp_counter
        name = f"TEMP{temp_counter}"
        temp_counter += 1
        return name

    for raw in parts:
        instr = raw + ";"
        if not re.match(r'^(U|V)\d+\s*=', instr.strip()):
            out_instrs.append(instr)
            continue

        left, right = instr.rstrip(';').split('=', 1)
        left = left.strip()
        rhs = right.strip()

        # Endogenous
        if left.startswith("V"):
            signed_terms = split_top_level_plus(rhs)
            pre_temps = []
            for sign, term in signed_terms:
                factors = split_factors(term)
                factors = reorder_number_first(factors)
                if len(factors) > 1 and any(is_varv(f) for f in factors):
                    temp = new_temp()
                    pre_temps.append((temp, factors, sign))
            for temp, factors, _ in pre_temps:
                out_instrs.append(f"{temp} = {factors[0]} * {factors[1]};")
                for f in factors[2:]:
                    out_instrs.append(f"{temp} = {temp} * {f};")
            first_term = True
            for sign, term in signed_terms:
                factors = split_factors(term)
                factors = reorder_number_first(factors)
                if len(factors) > 1 and any(is_varv(f) for f in factors):
                    temp_name = next(t for t, fcts, s in pre_temps if fcts == factors)
                    if first_term:
                        if sign == '+': out_instrs.append(f"{left} = {temp_name};")
                        else:
                            out_instrs.append(f"{left} = 0;")
                            out_instrs.append(f"{left} = {left} - {temp_name};")
                    else:
                        if sign == '+': out_instrs.append(f"{left} = {left} + {temp_name};")
                        else: out_instrs.append(f"{left} = {left} - {temp_name};")
                else:
                    if first_term:
                        if sign == '+': out_instrs.append(f"{left} = {term};")
                        else:
                            out_instrs.append(f"{left} = 0;")
                            out_instrs.append(f"{left} = {left} - {term};")
                    else:
                        if sign == '+': out_instrs.append(f"{left} = {left} + {term};")
                        else: out_instrs.append(f"{left} = {left} - {term};")
                first_term = False
            continue

        # Exogenous
        signed_terms = split_top_level_plus(rhs)
        temp_terms = []
        for sign, term in signed_terms:
            factors = split_factors(term)
            factors = reorder_number_first(factors)
            # Each term at most one multiplication
            if len(factors) > 1:
                temp = new_temp()
                out_instrs.append(f"{temp} = {factors[0]} * {factors[1]};")
                temp_str = f"{sign}{temp}" if sign == '-' else temp
                temp_terms.append(temp_str)
            else:
                temp_terms.append(f"{sign}{factors[0]}" if sign == '-' else factors[0])
        # Build target assignment with proper signs
        first = True
        expr = ""
        for t in temp_terms:
            if first:
                if t.startswith('-'):
                    expr = f"0 - {t[1:]}"
                else:
                    expr = t
                first = False
            else:
                if t.startswith('-'):
                    expr += f" - {t[1:]}"
                else:
                    expr += f" + {t}"
        out_instrs.append(f"{left} = {expr};")

    return "".join(out_instrs)
