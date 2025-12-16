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
    else:
        raise ValueError(f"Unknown process name: {process_name}")
    
## CASUAL SKILLS ##

SCM_causal_skills = {
    "skill": lambda: sample_V1_causal_skills(),
    "belief": lambda skill: sample_V2_causal_skills(skill),
    "result": lambda skill, belief: sample_V3_causal_skills(skill, belief),
}

ORDER_causal_skills = ["skill", "belief", "result"]

def sample_V1_causal_skills():
    return np.random.normal(50, 10)

def sample_V2_causal_skills(skill):
    return 0.5 * skill + np.random.normal(5, 2)

def sample_V3_causal_skills(skill, belief):
    return 0.3 * skill + 0.7 * belief + np.random.normal(3, 1)