from random import uniform
import numpy as np

def get_vars(process_name):
    if process_name == 'bernoulli':
        data_var_list = ['y']
        return data_var_list
    if process_name == 'test':
        data_var_list = ['y']
        return data_var_list
    if process_name == 'test2':
        data_var_list = ['y']
        return data_var_list
    if process_name == 'clickgraph':
        data_var_list = ['click0', 'click1']
        return data_var_list
    if process_name == "grass":
        data_var_list = ['rain', 'sprinkler', 'wetGrass', 'wetRoof']
        return data_var_list
    if process_name == 'murdermistery':
        data_var_list = ['withGun']
        return data_var_list
    if process_name == 'twocoins':
        data_var_list = ['both']
        return data_var_list
    if process_name == 'surveyunbiased':
        data_var_list = ['ansb1', 'ansb2']
        return data_var_list
    if process_name == 'trueskills':
        data_var_list = ['perfA', 'perfB', 'perfC']
        return data_var_list
    if process_name == 'altermu':
        data_var_list = ['y']
        return data_var_list
    if process_name == 'altermu2':
        data_var_list = ['y']
        return data_var_list
    if process_name == 'normalmixtures':
        data_var_list = ['y']
        return data_var_list
    if process_name =="noisior":
        data_var_list = ['n2', 'n3']
        return data_var_list
    if process_name == "pid":
        # variables are ang[1], ang[2], ..., ang[50]
        data_var_list = ['ang[{}]'.format(i) for i in range(1, 51)]
        return data_var_list
    if process_name == 'mog1':
        data_var_list = ['mu', 'sigma', 'x']
        dependencies = {'x': ['mu', 'sigma']}
        weights = {'x': 0.1}  # Weights for the dependencies
        return data_var_list, dependencies, weights   
    if process_name == 'burglary':
        data_var_list = ['burglary', 'earthquake', 'alarm', 'maryWakes', 'phoneWorking', 'called']
        return data_var_list   
    if process_name == 'clinicaltrial':
        data_var_list = ['ycontr', 'ytreated']
        return data_var_list
    if process_name == 'coinbias':
        data_var_list = ['y']
        return data_var_list
    if process_name == 'csi':
        data_var_list = ['u', 'v', 'w', 'x']
        dependencies = {'x': ['u', 'v', 'w']}   
        weights = {'x': 0.1}
        return data_var_list, dependencies, weights
    if process_name == 'healthiness':
        data_var_list = ['healthconscius', 'littlefreetime', 'exercise', 'gooddiet', 'normalweight', 'colesterol', 'tested']
        dependencies = {'exercise': ['healthconscius', 'littlefreetime'], 'gooddiet':['healthconcius'], 'normalweight': ['gooddiet', 'exercise'], 'colesterol': ['gooddiet'], 'tested': ['colesterol']}
        weights = {'exercise':0.1, 'gooddiet':0.1, 'normalweight':0.1, 'colesterol':0.1, 'tested':0.1}
        return data_var_list, dependencies, weights 
    if process_name == 'eyecolor':
        data_var_list = ['eyecolor', 'haircolor', 'hairlength']
        dependencies = {'haircolor': ['eyecolor']}
        weights = {}
        return data_var_list, dependencies, weights
    if process_name == 'easytugwar':
        data_var_list = ['skill1', 'skill2', 'p1wins']
        dependencies = {'p1wins': ['skill1', 'skill2']}
        weights = {'p1wins': 0.1} 
        return data_var_list, dependencies, weights    
    if process_name == 'hurricane':
        data_var_list = ['preplevel', 'damage']
        dependencies = {'damage': ['preplevel']}
        weights = {'damage': 0.1}
        return data_var_list, dependencies, weights
    if process_name == 'icecream':
        data_var_list = ['currentseason', 'icecream', 'crime']
        dependencies = {'icecream': ['currentseason'], 'crime': ['currentseason']}
        weights = {'icecream': 0.1, 'crime': 0.1}
        return data_var_list, dependencies, weights
    if process_name == 'biasedtugwar':
        data_var_list = ['skill1', 'skill2', 'p1wins']
        dependencies = {'p1wins': ['skill1', 'skill2']}
        weights = {'p1wins': 0.1} 
        return data_var_list, dependencies, weights
    if process_name == 'if':
        data_var_list = ['a', 'b']
        dependencies = {'b': ['a']}
        weights = {'b': 0.1} 
        return data_var_list, dependencies, weights
    if process_name == 'tugwaraddition':
        data_var_list = ['skill1', 'skill2', 'skill3' 'p1wins']
        dependencies = {'p1wins': ['skill1', 'skill2', 'skill3']}
        weights = {'p1wins': 0.1} 
        return data_var_list, dependencies, weights  
    if process_name == 'mixedcondition':
        data_var_list = ['u', 'v', 'w']
        dependencies = {'w': ['u', 'v']}
        weights = {'w': 0.1} 
        return data_var_list, dependencies, weights     
    if process_name == 'multiplebranches':
        data_var_list = ['contentDifficulty', 'questionsAfterLectureLength']
        dependencies = {'questionsAfterLectureLength': ['contentDifficulty']}
        weights = {'questionsAfterLectureLength': 0.1}
        return data_var_list, dependencies, weights
    else:
        raise ValueError(f"Unknown process name: {process_name}")
    
def get_params(process_name):
    if process_name == 'bernoulli':
        true_params = {'p': 0.8}
        init_params = {'p': 0.5}
        lr = 0.01
        lr_VI = 0.05
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'burglary':
        true_params = {'pe': 0.01, 'pb': 0.1}
        init_params = {'pe': 0.5, 'pb': 0.5}
        lr = 0.005
        lr_VI = 0.05
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'clickgraph':
        true_params = {'p': 0.8}
        init_params = {'p': 0.5}
        lr = 0.2
        lr_VI = 0.001
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'clinicaltrial':
        true_params = {'pe': 0.1, 'pc': 0.3, 'pt': 0.8}
        init_params = {'pe': 0.5, 'pc': 0.5, 'pt': 0.5}
        lr = 0.001
        lr_VI = 0.2
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'coinbias':
        true_params = {'p1': 2.0, 'p2': 5.0}
        init_params = {'p1': 1.0, 'p2': 1.0}
        lr = 0.01
        lr_VI = 0.05
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'grass':
        true_params = {'pcloudy': 0.5, 'p1': 0.7, 'p2': 0.9, 'p3': 0.9}
        init_params = {'pcloudy': 0.5, 'p1': 0.5, 'p2': 0.5, 'p3': 0.5}
        lr = 0.1
        lr_VI = 0.01
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'murdermistery':
        true_params = {'palice': 0.1}
        init_params = {'palice': 0.5}
        lr = 0.01
        lr_VI = 0.001
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'noisior':
        true_params = {'p0': 0.5, 'p1': 0.8, 'p2': 0.1, 'p4': 0.5}
        init_params = {'p0': 0.3, 'p1': 0.3, 'p2': 0.3, 'p4': 0.3}
        lr = 0.01
        lr_VI = 0.001
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'surveyunbiased':
        true_params = {'bias1': 0.8, 'bias2': 0.7}
        init_params = {'bias1': 0.5, 'bias2': 0.5}
        lr = 0.01
        lr_VI = 0.01
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == "trueskills":
        true_params = {'pa': 105., 'pb': 90., 'pc': 120.}
        init_params = {'pa': 100., 'pb': 100., 'pc': 100.}
        lr = 0.2
        lr_VI = 0.2
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'twocoins':
        true_params = {'first': 0.8, 'second': 0.2}
        init_params = {'first': 0.5, 'second': 0.5}
        lr = 0.001
        lr_VI = 0.05
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'altermu':
        true_params = {'p1': 1.0, 'p2': 2.0, 'p3': 3.0}
        init_params = {'p1': 0.5, 'p2': 0.5, 'p3': 0.5}
        lr = 0.05
        lr_VI = 0.0005
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'altermu2':
        true_params = {'muy': 3.0, 'vary': 3.0}
        init_params = {'muy': 0.0, 'vary': 1.0}
        lr = 0.2
        lr_VI = 0.001
        mcmc_steps = 1500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'normalmixtures':
        true_params = {'theta': 0.7, 'p1': -10.0, 'p2': 1.0}
        init_params = {'theta': 0.5, 'p1': 0.0, 'p2': 0.0}
        lr = 0.01
        lr_VI = 0.0005
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'test':
        true_params = {'p1': 0.5, 'p2': 1.0}
        init_params = {'p1': 0., 'p2': 0.}
        lr = 0.01
        lr_VI = 0.001
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'test2':
        true_params = {'p1': 1.0, 'sigma1': 2.0,}
        init_params = {'p1': 0., 'sigma1': 1.0}
        lr = 0.01
        lr_VI = 0.001
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup
    if process_name == 'pid':
        true_params = {'s0': 46., 's1': -23., 's2': 0.001}
        init_params = {'s0': 46., 's1': -23., 's2': 0.001}
        lr = 0.2
        lr_VI = 0.05
        mcmc_steps = 500
        mcmc_warmup = 50
        return true_params, init_params, lr, lr_VI, mcmc_steps, mcmc_warmup

def generate_dataset(process_name, data_size, params):
    if process_name == 'bernoulli':
        return generate_bernoulli_dataset(data_size, params)
    if process_name == 'clickgraph':
        return generate_clickgraph_dataset(data_size, params)
    if process_name == 'clinicaltrial':
        return generate_clinicaltrial_dataset(data_size, params)
    if process_name == 'coinbias':
        return generate_coinbias_dataset(data_size, params)
    if process_name == 'grass':
        return generate_grass_dataset(data_size, params)
    if process_name == 'trueskills':
        return generate_trueskills_dataset(data_size, params)
    if process_name == 'altermu':
        return generate_altermu_dataset(data_size, params)
    if process_name == 'altermu2':
        return generate_altermu2_dataset(data_size, params)
    if process_name == 'normalmixtures':
        return generate_normalmixtures_dataset(data_size, params)
    if process_name == 'murdermistery':
        return generate_murdermistery_dataset(data_size, params)
    if process_name == 'noisior':
        return generate_noisior_dataset(data_size, params)
    if process_name == 'twocoins':
        return generate_twocoins_dataset(data_size, params)
    if process_name == 'surveyunbiased':
        return generate_surveyunbiased_dataset(data_size, params)
    if process_name == 'mog1':
        return generate_mog1_dataset(data_size)
    if process_name == 'burglary':
        return generate_burglary_dataset(data_size, params)
    if process_name == 'csi':
        return generate_csi_dataset(data_size)
    if process_name == 'healthiness':  
        return generate_healthiness_dataset(data_size)
    if process_name == 'eyecolor':
        return generate_eyecolor_dataset(data_size)
    if process_name == 'easytugwar':
        return generate_easytugwar_dataset(data_size)
    if process_name == 'hurricane':
        return generate_hurricane_dataset(data_size)
    if process_name == 'icecream':
        return generate_icecream_dataset(data_size)
    if process_name == 'biasedtugwar':
        return generate_biasedtugwar_dataset(data_size)
    if process_name == 'if':
        return generate_if_dataset(data_size)
    if process_name == 'tugwaraddition':
        return generate_biasedtugwar_dataset(data_size)
    if process_name == 'mixedcondition':
        return generate_mixedcondition_dataset(data_size)
    if process_name == 'multiplebranches':
        return generate_multiplebranches_dataset(data_size)
    if process_name == 'test':
        return generate_test_dataset(data_size, params)
    if process_name == 'test2':
        return generate_test2_dataset(data_size, params)
    if process_name == 'pid':
        return generate_pid_dataset(data_size, params)
    else:
        raise ValueError(f"Unknown process name: {process_name}")
    
def generate_bernoulli_dataset(data_size, params):
    p = params['p']
    data = []
    for _ in range(data_size):
        y = np.random.binomial(1, p)
        data.append([y])
    return data

def generate_trueskills_dataset(data_size, params):
    data = []
    pa = params['pa']
    pb = params['pb']
    pc = params['pc']
    for _ in range(data_size):
        skillA = np.random.normal(pa, 10)
        skillB = np.random.normal(pb, 10)
        skillC = np.random.normal(pc, 10)
        perfA = np.random.normal(0, 15) + skillA
        perfB = np.random.normal(0, 15) + skillB
        perfC = np.random.normal(0, 15) + skillC
        data.append([perfA, perfB, perfC])
    return data

def generate_mog1_dataset(data_size):
    data = []
    for _ in range(data_size):
        mu = np.random.normal(20, 3)
        sigma = np.random.normal(2, 1)
        x = mu + sigma * np.random.normal(1, 1)
        data.append([mu, sigma, x])
    return data

def generate_clickgraph_dataset(data_size, params):
    p= params['p']
    data = []
    for _ in range(data_size):
        sim = np.random.binomial(1, p)
        beta1 = np.random.uniform(0, 1)
        if sim == 1:
            beta2 = beta1
        else:
            beta2 = np.random.uniform(0, 1)
        click0 = np.random.binomial(1, beta1)
        click1 = np.random.binomial(1, beta2)
        data.append([click0, click1])
    return data

def generate_burglary_dataset(data_size, params):
    pe, pb = params['pe'], params['pb']
    data = []
    for _ in range(data_size):
        burglary = np.random.binomial(1, pb)  
        earthquake = np.random.binomial(1, pe) 
        if burglary:
            alarm = 1.
        else:
            if earthquake:
                alarm = 1.
            else:
                alarm = 0.
        if alarm:
            if earthquake:
                maryWakes = np.random.binomial(1, 0.8)
            else:
                maryWakes = np.random.binomial(1, 0.6)
        else:
            maryWakes = np.random.binomial(1, 0.2)

        if earthquake:
            phoneWorking = np.random.binomial(1, 0.7)
        else:
            phoneWorking = np.random.binomial(1, 0.99)
        if maryWakes:
            if phoneWorking:
                called = 1.
            else:
                called = 0.
        else:
            called = 0.

        data.append([burglary, earthquake, alarm, maryWakes, phoneWorking, called])
    return data

def generate_csi_dataset(data_size):
    data = []
    for _ in range(data_size):
        u = np.random.binomial(1, 0.3)
        v = np.random.binomial(1, 0.9)
        w = np.random.binomial(1, 0.1)
        if u:
            if w:
                x = np.random.binomial(1, 0.8)
            else:
                x = np.random.binomial(1, 0.2)
        else:
            if v:
                x = np.random.binomial(1, 0.8)
            else:
                x = np.random.binomial(1, 0.2)
        data.append([u, v, w, x])
    return data

def generate_healthiness_dataset(data_size):
    data = []
    for _ in range(data_size):
        healthconscius = np.random.binomial(1, 0.5)
        littlefreetime = np.random.binomial(1, 0.5)
        if healthconscius:
            if littlefreetime:
                exercise = np.random.binomial(1, 0.5)
            else:
                exercise = np.random.binomial(1, 0.9)
        else:
            if littlefreetime:
                exercise = np.random.binomial(1, 0.1)
            else:
                exercise = np.random.binomial(1, 0.5)
        
        if healthconscius:
            gooddiet = np.random.binomial(1, 0.7)
        else:
            gooddiet = np.random.binomial(1, 0.3)
        
        if gooddiet:
            if exercise:
                normalweight = np.random.binomial(1, 0.8)
            else:
                normalweight = np.random.binomial(1, 0.5)
        else:
            if exercise:
                normalweight = np.random.binomial(1, 0.5)
            else:
                normalweight = np.random.binomial(1, 0.2)
        
        if gooddiet:
            colesterol = np.random.binomial(1, 0.3)
        else:
            colesterol = np.random.binomial(1, 0.7)
        
        if colesterol:
            tested = np.random.binomial(1, 0.9)
        else:
            tested = np.random.binomial(1, 0.1)
        data.append([healthconscius, littlefreetime, exercise, gooddiet, normalweight, colesterol, tested])
    return data

def generate_eyecolor_dataset(data_size):
    data = []
    for _ in range(data_size):
        eyecolor = np.random.choice([0, 1, 2, 3], p=[0.82, 0.08, 0.08, 0.02])
        if eyecolor == 0:
            haircolor = np.random.choice([0, 1, 2, 3, 4], p=[0.8, 0.05, 0.04, 0.01, 0.1])
        elif eyecolor == 1:
            haircolor = np.random.choice([0, 1, 2, 3, 4], p=[0.7, 0.15, 0.04, 0.01, 0.1])
        elif eyecolor == 2:
            haircolor = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.18, 0.02, 0.1])
        else:
            haircolor = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.29, 0.18, 0.03, 0.1])
        hairlength = np.random.choice([0, 1, 2], p=[0.6, 0.15, 0.25])
        data.append([eyecolor, haircolor, hairlength])
    return data

def generate_easytugwar_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(20, 4)
        skill2 = np.random.normal(20, 4)
        if skill1 > skill2:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data


def generate_hurricane_dataset(data_size):
    data = []
    for _ in range(data_size):
        preplevel = np.random.choice([0, 1, 2], p=[0.5, 0.2, 0.3])
        if preplevel == 0:
            damage = np.random.choice([0,1], p=[0.2, 0.8])
        elif preplevel == 1:
            damage = np.random.choice([0,1], p=[0.2, 0.8])
        elif preplevel == 2:
            damage = np.random.choice([0,1], p=[0.8, 0.2])
        data.append([preplevel, damage])
    return data

def generate_icecream_dataset(data_size):
    data = []
    for _ in range(data_size):
        currentseason = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
        if currentseason == 0:
            icecream = np.random.normal(10, 1)
            crime = np.random.normal(10, 1)
        elif currentseason == 1:
            icecram = np.random.normal(15, 3)
            crime = np.rancom.normal(15, 3)
        elif currentseason == 2:
            icecream = np.random.normal(50, 6)
            crime = np.random.normal(50, 6)
        elif currentseason == 3:
            icecram = np.random.normal(17, 4)
            crime = np.random.normal(17, 4)
        data.append([currentseason, icecream, crime])
    return data

def generate_biasedtugwar_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(20, 4)
        skill2 = np.random.normal(20, 4)
        if skill1 > 1.3 * skill2:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data

def generate_if_dataset(data_size):
    data = []
    for _ in range(data_size):
        a = np.random.normal(1, 2)
        if a < 0:
            b = a * 3 + np.random.normal(0, 1)
        else:
            b = np.random.normal(8, 1)
        data.append([a, b])
    return data

def generate_tugwaraddition_dataset(data_size):
    data = []
    for _ in range(data_size):
        skill1 = np.random.normal(38, 6)
        skill2 = np.random.normal(20, 4)
        skill3 = np.random.normal(20, 4)
        if skill1 > skill2 + skill3:
            p1wins = 1.0
        else:
            p1wins = 0.0
        data.append([skill1, skill2, p1wins])
    return data

def generate_mixedcondition_dataset(data_size):
    data = []
    for _ in range(data_size):
        u = np.random.binomial(1, 0.3)
        v = np.random.normal(10, 2)
        if u and v > 12.0:
            w = np.random.normal(12, 2)
        else:
            w = np.random.normal(6, 2)
        data.append([u, v, w])
    return data

def generate_multiplebranches_dataset(data_size):
    data = []
    for _ in range(data_size):
        contentDifficulty = np.random.normal(30, 5)
        if contentDifficulty < 35:
            if contentDifficulty < 20:
                questionsAfterLectureLength = np.random.normal(2, 1)
            else:
                questionsAfterLectureLength = np.random.normal(10, 3)
        else:
            questionsAfterLectureLength = np.random.normal(25, 6)
        data.append([contentDifficulty, questionsAfterLectureLength])

    return data

def generate_clinicaltrial_dataset(data_size, params):
    data = []
    pe, pc, pt = params['pe'], params['pc'], params['pt']
    for _ in range(data_size):
        effect = np.random.binomial(1, pe)
        if effect:
            pc = pt
        ycontr = np.random.binomial(1, pc)
        ytreated = np.random.binomial(1, pt)
        data.append([ycontr, ytreated])
    return data

def generate_coinbias_dataset(data_size, params):
    p1, p2 = params['p1'], params['p2']
    data = []
    for _ in range(data_size):
        bias = np.random.beta(p1, p2)
        y = np.random.binomial(1, bias)
        data.append([y])
    return data

def generate_grass_dataset(data_size, params):
    data = []
    pcloudy, p1, p2, p3 = params['pcloudy'], params['p1'], params['p2'], params['p3']
    for _ in range(data_size):
        cloudy = np.random.binomial(1, pcloudy)
        if cloudy == 1:
            rain = np.random.binomial(1, 0.8)       
            sprinkler = np.random.binomial(1, 0.1)  
        else:
            rain = np.random.binomial(1, 0.2)       
            sprinkler = np.random.binomial(1, 0.5)  

        temp1 = np.random.binomial(1, p1)  
        if temp1 == 1:
            wetRoof = 1 if rain == 1 else 0
        else:
            wetRoof = 0

        temp2 = np.random.binomial(1, p2) 
        temp3 = np.random.binomial(1, p3)  

        if temp2 == 1:
            or1 = 1 if rain == 1 else 0
        else:
            or1 = 0

        if temp3 == 1:
            or2 = 1 if sprinkler == 1 else 0
        else:
            or2 = 0

        if or1 == 0:
            if or2 == 0:
                wetGrass = 0
            else:
                wetGrass = 1
        else:
            wetGrass = 1
        data.append([rain, sprinkler, wetGrass, wetRoof])
    return data

def generate_murdermistery_dataset(data_size, params):
    data = []
    palice = params['palice']
    for _ in range(data_size):
        alice = np.random.binomial(1, palice)
        if alice:
            withGun = np.random.binomial(1, 0.03)
        else:
            withGun = np.random.binomial(1, 0.8)
        data.append([withGun])
    return data

def generate_noisior_dataset(data_size, params):
    data = []
    p0, p1, p2, p4 = params['p0'], params['p1'], params['p2'], params['p4']
    for _ in range(data_size):
        n0 = np.random.binomial(1, p0)
        n4 = np.random.binomial(1, p4)
        if n0 == 1:
            n1 = p1
            n21 = p1
        else:
            n1 = p2
            n21 = p2

        if n4 == 1:
            n22 = p1
            n33 = p1
        else:
            n22 = p2
            n33 = p2

        if np.random.binomial(1, n21) == 0:
            if np.random.binomial(1, n22) == 0:
                n2 = 0
            else:
                n2 = 1
        else:
            n2 = 1

        if np.random.binomial(1, n1) == 1:
            n31 = p1
        else:
            n31 = p2
        
        if n2 == 1:
            n32 = p1
        else:
            n32 = p2
        if np.random.binomial(1, n31) == 0:
            if np.random.binomial(1, n32) == 0:
                if np.random.binomial(1, n33) == 0:
                    n3 = 0
                else:
                    n3 = 1
            else:
                n3 = 1
        else:
            n3 = 1
        data.append([n2, n3])
    return data

def generate_surveyunbiased_dataset(data_size, params):
    data = []
    bias1, bias2 = params['bias1'], params['bias2']
    for _ in range(data_size):
        ansb1 = np.random.binomial(1, bias1)
        ansb2 = np.random.binomial(1, bias2)
        data.append([ansb1, ansb2])
    return data

def generate_twocoins_dataset(data_size, params):
    data = []
    first, second = params['first'], params['second']
    for _ in range(data_size):
        coin1 = np.random.binomial(1, first)
        coin2 = np.random.binomial(1, second)
        if coin1 == 1 and coin2 == 1:
            both = 1
        else:
            both = 0
        data.append([both])
    return data

def generate_altermu_dataset(data_size, params):
    data = []
    p1, p2, p3 = params['p1'], params['p2'], params['p3']
    for _ in range(data_size):
        w1 = np.random.normal(p1, 5.)
        w2 = np.random.normal(p2, 5.)
        w3 = np.random.normal(p3, 5.)
        mean = w1 * w2
        mean = 3 * mean - w3
        y = np.random.normal(mean, 1.)
        data.append([y])
    return data

def generate_altermu2_dataset(data_size, params):
    data = []
    muy, vary = params['muy'], params['vary']
    for _ in range(data_size):
        w1 = np.random.uniform(-10, 10)
        w2 = np.random.uniform(-10, 10)
        y = w1 + w2 + np.random.normal(muy, vary)
        data.append([y])
    return data

def generate_normalmixtures_dataset(data_size, params):
    data = []
    theta, p1, p2 = params['theta'], params['p1'], params['p2']
    for _ in range(data_size):
        mu1 = np.random.normal(p1, 1)
        mu2 = np.random.normal(p2, 1)
        if np.random.binomial(1, theta) == 1:
            y = np.random.normal(mu1, 1)
        else:
            y = np.random.normal(mu2, 1)
        data.append([y])
    return data

def generate_test_dataset(data_size, params):
    data = []
    p1, p2 = params['p1'], params['p2']
    for _ in range(data_size):
        v = np.random.normal(p1, 5)
        if v > 0:
            y = np.random.normal(p2, 1)
        else:
            y = np.random.normal(-2, 1)
        data.append([y])
    return data

def generate_test2_dataset(data_size, params):
    data = []
    p1, sigma1 = params['p1'], params['sigma1']
    for _ in range(data_size):
        v = np.random.normal(0., sigma1)
        if v < p1:
            y = 0.
        else:
            y = 1.
        data.append([y])
    return data

def generate_pid_dataset(data_size, params):
    data = []
    s0, s1, s2 = params['s0'], params['s1'], params['s2']
    for _ in range(data_size):
        target = 3.14
        T = 50
        '''
        dt = 0.1
        inertia = 10
        decay = 0.9
        
        init_ang = 0.5
        
        traj = []
        noise = np.random.normal(0, 0.25)
        noise_ang = np.random.normal(0, 0.25)

        v = 0 
        ang = init_ang 
        id = 0 

        for i in range(0,T):

            traj.append(ang)

            d = target - ang
            torq = s0*d + s1*v + s2*id
            id = decay*id + d*dt
            oldv = v 
            v = v + (dt/inertia)*torq + noise.rsample()
            ang = ang + (dt/2)*(v+oldv) + noise_ang.rsample()

        traj[T-1] = ang 
        '''
        ang = [target for _ in range(T)]
        data.append(ang)
    return data