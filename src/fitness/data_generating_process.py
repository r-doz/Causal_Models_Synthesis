import numpy as np

def generate_interventional_dataset(SCM, order, data_size, intervention=None):
    """
    Generate a dataset of samples under an intervention.
    """
    data = []
    for _ in range(data_size):
        sample = sample_scm_with_intervention(SCM, order, intervention)
        data.append([sample[var] for var in order])
    return np.array(data)

def get_intervention_list(program: str):
    if program == 'chain':
        return [("F", 5.0), ("F", 15.0),("T", 5.0), ("T", 20.0), ("F", 1.0), ("T", 1.0), ("F", 8.5), ("P", 50.0), ("P", 150.0), ("P", 500.0)]


def sample_scm_with_intervention(SCM, order, intervention=None):
    """
    Sample a single SCM instance with optional intervention.
    
    intervention: dict or None, e.g., {"V2": 10.0}
    """
    if intervention is None:
        intervention = {}

    values = {}

    for var in order:
        if var in intervention:
            # Apply intervention (do-operator)
            values[var] = intervention[var]
        else:
            # Compute variable using SCM structural function
            f = SCM[var]
            # Collect required parent values
            parents = {p: values[p] for p in f.__code__.co_varnames}
            values[var] = f(**parents)

    return values


def get_vars(process_name):
    if process_name == 'causal_skills':
        return ORDER_causal_skills, SCM_causal_skills
    if process_name == 'chain':
        return ORDER_chain, SCM_chain
    if process_name == 'common_cause':
        return ORDER_common_cause, SCM_common_cause
    if process_name == 'common_effect':
        return ORDER_common_effect, SCM_common_effect
    if process_name == 'diamond':
        return ORDER_diamond, SCM_diamond
    if process_name == 'complex':
        return ORDER_complex, SCM_complex
    else:
        raise ValueError(f"Unknown process name: {process_name}")
    
## CASUAL SKILLS ##

SCM_causal_skills = {
    "skill": lambda: sample_V1_causal_skills(),
    "belief": lambda skill: sample_V2_causal_skills(skill),
    "result": lambda skill, belief: sample_V3_causal_skills(skill, belief),
}

SCM_chain = {
    "F": lambda: np.random.normal(10, 2),
    "T": lambda F: sample_T_chain(F) + np.random.normal(0,1),
    "P": lambda T: T*T + np.random.normal(0, 10),
}

SCM_common_cause = {
    "F": lambda: np.random.normal(10, 2),
    "C": lambda F: 0.5*F + np.random.uniform(-1,1),
    "T": lambda F: sample_T_chain(F)+ np.random.normal(0,1),
}

SCM_common_effect = {
    "F": lambda: np.random.normal(10, 2),
    "W": lambda: np.random.uniform(0,10),
    "T": lambda F, W: sample_T_chain(F) + W + np.random.normal(0,1),
}

SCM_diamond = {
    "F": lambda: np.random.normal(10, 2),
    "C": lambda F: 0.5*F + np.random.uniform(-1,1),
    "T": lambda F: sample_T_chain(F) + np.random.normal(0,1),
    "E": lambda C, T: 50 * C + T * T + np.random.normal(0,10),
}
SCM_complex = {
    "F": lambda: np.random.normal(10, 2),
    "C": lambda F: 0.5*F + np.random.uniform(-1,1),
    "W": lambda: np.random.uniform(0,10),
    "T": lambda F, W: sample_T_chain(F) + W + np.random.normal(0,1),
    "P": lambda T: T*T + np.random.normal(0, 10),
}

ORDER_causal_skills = ["skill", "belief", "result"]
ORDER_chain = ["F", "T", "P"]
ORDER_common_cause = ["F", "C", "T"]
ORDER_common_effect = ["F", "W", "T"]
ORDER_diamond = ["F", "C", "T", "E"]
ORDER_complex = ["F", "C", "W", "T", "P"]

def sample_V1_causal_skills():
    return np.random.normal(30, 5)

def sample_V2_causal_skills(skill):
    return 0.5 * skill + np.random.normal(5, 2)

def sample_V3_causal_skills(skill, belief):
    return 0.3 * skill + 5 * belief + np.random.normal(3, 0.1)

def sample_T_chain(F):
    if F < 8.5:
        return F
    else: 
        return 20
    

def get_real_program(program):
    if program == 'chain':
        code = """
        F = gauss(10, 2);
        if F < 8.5 {
            T = F;
        } else {
            T = 20; 
        } end if;
        T = T + gauss(0, 1);
        P = T * T;
        P = P + gauss(0, 10);
        """
    return code

def get_baseline_program(program):
    if program == 'chain':
        code = """
        F = gauss(10, 2);
        T = gauss(16.75, 5.67);
        P = gauss(312.56, 156.30);   
        """
    return code