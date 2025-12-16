# Contains the functions for computing the resulting distribution when a truncation occurs in conditional or observe instructions according to the following dependencies.

from libSOGAshared import *
from TRUNCLexer import *
from TRUNCParser import *
from TRUNCListener import *
import timing
import multiprocessing as mp
from functools import partial

pool=None

def ineq_func(self,comp):
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    for part in product(*[range(len(mean)) for mean in self.aux_means]):
        # for a given combination of components of the auxiliary variables, creates a new component extending comp
        aux_pi = 1
        aux_mean = list(copy(mu))
        aux_sigma = []
        ineq_coeff = np.array(copy(self.coeff))
        ineq_const = self.const
        for p,q in zip(range(len(self.aux_means)), part):
            aux_pi = aux_pi*self.aux_pis[p][q]
            aux_mean.append(self.aux_means[p][q])
            aux_sigma.append(self.aux_covs[p][q])
        aux_mean = np.array(aux_mean)
        aux_sigma = np.diag(aux_sigma)
        aux_cov = np.block([[sigma, np.zeros((len(sigma), len(aux_sigma)))], [np.zeros((len(aux_sigma), len(sigma))), aux_sigma]])
        # substitute deltas
        delta_idx = np.where(np.diag(aux_cov) < delta_tol)[0]
        ineq_const -= np.array(self.coeff)[delta_idx].dot(aux_mean[delta_idx])
        ineq_coeff[delta_idx] = np.zeros(len(delta_idx))
        # if all variables were deltas return
        if np.all(np.array(ineq_coeff) == 0):
            if (self.type == '>' and ineq_const < 0) or (self.type == '>=' and ineq_const <= 0) or (self.type == '<' and ineq_const > 0) or (self.type == '<=' and ineq_const >= 0):
                new_P = 1.
            else:
                new_P = 0.
            new_mu = mu
            new_sigma = sigma
        # else compute truncated distribution
        else:
            # STEP 1: change variables
            norm = np.linalg.norm(ineq_coeff)
            ineq_coeff = np.array(ineq_coeff)/norm
            ineq_const = ineq_const/norm
            A = find_basis(ineq_coeff)           # maybe instead of A a vector can be used to improve scalability (?)
            transl_mu = A.dot(aux_mean)
            transl_sigma = A.dot(aux_cov).dot(A.transpose())
            # STEP 2: finds the indices of the components that needs to be transformed
            transl_alpha = np.zeros(len(transl_mu))
            transl_alpha[0] = 1
            indices = select_indices(transl_alpha, transl_sigma)
            # STEP 3: creates reduced vectors taking into account only the coordinates that need to be transformed
            red_transl_alpha = reduce_indices(transl_alpha, indices)
            red_transl_mu = reduce_indices(transl_mu, indices)
            red_transl_sigma = reduce_indices(transl_sigma, indices) 
            # STEP 4: creates the hyper-rectangle to integrate on
            a = np.ones(len(red_transl_alpha))*(-np.inf)
            b = np.ones(len(red_transl_alpha))*(np.inf)
            if self.type=='>' or self.type=='>=':
                a[0] = ineq_const
            if self.type=='<' or self.type=='<=':
                b[0] = ineq_const   
            # STEP 5: compute moments in the transformed coordinates
            new_P, new_red_transl_mu, new_red_transl_sigma = compute_moments(red_transl_mu, red_transl_sigma, a, b)
            # STEP 6: recreates extended vectors
            new_transl_mu = extend_indices(new_red_transl_mu, transl_mu, indices)
            new_transl_sigma = extend_indices(new_red_transl_sigma, transl_sigma, indices)
            # STEP 7: goes back to older coordinates
            d = len(comp.var_list)
            A_inv = np.linalg.inv(A)
            new_mu = A_inv.dot(new_transl_mu)[:d]
            new_sigma = A_inv.dot(new_transl_sigma).dot(A_inv.transpose())[:d,:d]
            end = time()
        # append new values
        final_pi.append(aux_pi*new_P)
        final_mu.append(new_mu)
        final_sigma.append(new_sigma)
    return GaussianMix(final_pi, final_mu, final_sigma)

def eq_func(self,comp):
    mu = comp.gm.mu[0]
    sigma = comp.gm.sigma[0]
    final_pi = []
    final_mu = []
    final_sigma = []
    eq_coeff = copy(self.coeff)
    eq_const = self.const
    # check if delta
    i = np.where(np.array(self.coeff) != 0)[0][0]
    if sigma[i,i] < delta_tol:
        eq_const = eq_const - self.coeff[i]*mu[i]
        eq_coeff[i] = 0.
    # if delta return
    if np.all(np.array(eq_coeff) == 0):
        if (self.type == '==' and eq_const == 0) or (self.type == '!=' and eq_const != 0):
            new_P = 1.
        else:
            new_P = 0.
        new_mu = mu
        new_sigma = sigma
    else:
        # STEP 1: selects indices to condition
        indices = select_indices(eq_coeff, sigma)
        if len(indices) == 1:
            new_P = 0.
            new_mu = mu
            new_sigma = sigma
        else:
            # STEP 2: creates reduced vectors
            red_mu = reduce_indices(mu, indices)
            red_sigma = reduce_indices(sigma, indices) 
            red_alpha = reduce_indices(eq_coeff, indices)
            red_obs_idx = int(list(np.where(np.array(red_alpha)!=0))[0][0])
            # STEP 3: computes cond_sigma (select is a mask containing the index of the conditioned variables)
            select = (np.arange(len(red_mu))!=red_obs_idx)
            cond_sigma = red_sigma[select,:][:,select]
            cond_sigma = cond_sigma - (1/red_sigma[red_obs_idx,red_obs_idx])*(red_sigma[select,red_obs_idx].reshape(len(select)-1,1)).dot(red_sigma[red_obs_idx,select].reshape(1,len(select)-1))
            # STEP 4: computes cond_mu
            cond_mu = red_mu[select] + (1/red_sigma[red_obs_idx,red_obs_idx])*(eq_const-red_mu[red_obs_idx])*red_sigma[select,red_obs_idx]
            # if conditioned matrix is Null, it is equivalent to observing a single independent component
            if np.all(cond_sigma == 0):  
                new_P = 0.
            else:
                new_P = 1.
            # STEP 5: adds value for the observed variable (now a delta)
            cond_mu, cond_sigma = insert_value(eq_const, red_obs_idx, cond_mu, cond_sigma)
            # STEP 6: returns to the original set of variables
            new_sigma = extend_indices(cond_sigma, sigma, indices)
            new_mu = extend_indices(cond_mu, mu, indices) 
    return GaussianMix([new_P], [new_mu], [new_sigma])

def negate(trunc):
    """ Produces a string which is the logic negation of trunc """
    if '<' in trunc:
        if '<=' in trunc:
            trunc = trunc.replace('<=', '>')
        else:
            trunc = trunc.replace('<', '>=')
    elif '>' in trunc:
        if '>=' in trunc:
            trunc = trunc.replace('>=', '<')
        else:
            trunc = trunc.replace('>', '<=')
    elif '==' in trunc:
        trunc = trunc.replace('==', '!=')
    elif '!=' in trunc:
        trunc = trunc.replace('!=', '==')
    return trunc

def split_trunc(trunc):
    """ When trunc is 'x != c' returns 'x > c' and 'x < c' """
    assert '!=' in trunc
    trunc1 = trunc.replace('!=','>')
    trunc2 = trunc.replace('!=','<')
    return trunc1, trunc2

class TruncRule(TRUNCListener):
    
    def __init__(self, var_list, data):
        self.var_list = var_list
        self.data = data
        self.type = None
        self.coeff = [0.]*len(var_list)
        self.const = 0
        self.func = None
        
        self.aux_pis = []
        self.aux_means = []
        self.aux_covs = []
        
    def enterIneq(self, ctx):
        self.type = ctx.inop().getText()
        if not ctx.const().NUM() is None:
            self.const = float(ctx.const().NUM().getText())
        elif not ctx.const().idd() is None:
            self.const = ctx.const().idd().getValue(self.data)
                
    
    def enterLexpr(self, ctx):
        self.flag_sign = 1.

            
    def exitLexpr(self, ctx):
        self.func = partial(ineq_func,self)
        
        
    def enterMonom(self,ctx):
        if ctx.var().gm() is None:
            # monom in the form const? '*' (IDV | idd)
            ID = ctx.var()._getText(self.data)
            if not ctx.const() is None:
                if not ctx.const().NUM() is None:
                    coeff = self.flag_sign*float(ctx.const().NUM().getText())
                elif not ctx.const().idd() is None:
                    coeff = self.flag_sign*ctx.const().idd().getValue(self.data)
            else:
                coeff = self.flag_sign
            idx = self.var_list.index(ID)
            self.coeff[idx] = coeff
        # monom in the form const? '*' gm
        else:
            self.aux_pis.append(eval(ctx.var().gm().list_()[0].getText()))
            self.aux_means.append(eval(ctx.var().gm().list_()[1].getText()))
            self.aux_covs.append(np.array(eval(ctx.var().gm().list_()[2].getText()))**2)
            if not ctx.const() is None:
                if not ctx.const().NUM() is None:
                    coeff = self.flag_sign*float(ctx.const().NUM().getText())
                elif not ctx.const().idd() is None:
                    coeff = self.flag_sign*ctx.const().idd().getValue(self.data)
            else:
                coeff = self.flag_sign 
            self.coeff.append(self.flag_sign)            
            
    def enterSub(self, ctx):
        self.flag_sign = -1.
        
    def enterSum(self, ctx):
        self.flag_sign = 1.
        
    def enterEq(self, ctx):
        self.type = ctx.eqop().getText()
        idx = self.var_list.index(ctx.var()._getText(self.data))
        self.coeff[idx] = 1.
        if not ctx.const() is None:
            if not ctx.const().NUM() is None:
                self.const = float(ctx.const().NUM().getText())
            elif not ctx.const().idd() is None:
                self.const = ctx.const().idd().getValue(self.data)
            
        self.func = partial(eq_func,self)


def truncate(dist, trunc, data):
    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution. """
    if trunc == 'true':
        return 1., dist
    elif trunc == 'false':
        return 0., dist
    else:
        trunc_rule = trunc_parse(dist.var_list, trunc, data)
        trunc_func = trunc_rule.func
        trunc_type = trunc_rule.type
        trunc_idx = np.where(np.array(trunc_rule.coeff) != 0)[0][0]
        hard = []
        new_dist = Dist(dist.var_list, GaussianMix([],[],[]))
        new_pi = [] 
        trans_comp = []
        for k in range(dist.gm.n_comp()):
            comp = Dist(dist.var_list, dist.gm.comp(k))  
            trans_comp.append(trunc_func(comp))
        for k in range(dist.gm.n_comp()):
            if trunc_type == '==' and dist.gm.sigma[k][trunc_idx,trunc_idx] < delta_tol and sum(trans_comp[k].pi) > 0:
                hard.append(k)
        if len(hard) == 0:
            for k in range(dist.gm.n_comp()):
                new_mix = trans_comp[k]
                for h in range(new_mix.n_comp()):
                    if new_mix.pi[h] > prob_tol:
                        new_dist.gm.mu.append(new_mix.mu[h])
                        new_dist.gm.sigma.append(new_mix.sigma[h])
                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        else:
            for k in hard:
                new_mix = trans_comp[k]
                for h in range(new_mix.n_comp()):
                    if new_mix.pi[h] > prob_tol:
                        new_dist.gm.mu.append(new_mix.mu[h])
                        new_dist.gm.sigma.append(new_mix.sigma[h])
                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        norm_factor = sum(np.array(new_pi))
        if norm_factor > prob_tol:
            new_dist.gm.pi = list(np.array(new_pi)/norm_factor)
        else:
            new_dist.gm.pi = [0.]
            new_dist.gm.mu = [dist.gm.mu[0]]
            new_dist.gm.sigma = [dist.gm.sigma[0]]
        return norm_factor, new_dist
        #if norm_factor < prob_tol:
        #    return 0, dist

# parallel implementation
def parallel_truncate(dist, trunc, data,nproc):
    global pool
    gst=time()
    if(pool is None):
        print("creating pool")
        #pool=ThreadPoolExecutor(max_workers=nproc)
        pool=mp.Pool(nproc)

    """ Given a distribution dist computes its truncation to trunc. Returns a pair norm_factor, new_dist where norm_factor is the probability mass of the original distribution dist on trunc and new_dist is a Dist object representing the (approximated) truncated distribution. """
    if trunc == 'true':
        return 1., dist
    elif trunc == 'false':
        return 0., dist
    else:
        #print(f"##### ncomp={dist.gm.n_comp()}")
        st=time()
        trunc_rule = trunc_parse(dist.var_list, trunc, data)
        #print(f"trunc_parse:{time()-st}")
        trunc_func = trunc_rule.func
        trunc_type = trunc_rule.type
        trunc_idx = np.where(np.array(trunc_rule.coeff) != 0)[0][0]
        hard = []
        new_dist = Dist(dist.var_list, GaussianMix([],[],[]))
        new_pi = []
        comp_list = []

        st=time()
        for k in range(dist.gm.n_comp()):
            comp = Dist(dist.var_list, dist.gm.comp(k))
            comp_list.append(comp)
        #print(f"loop time:{time()-st}")

        st=time()

        trans_comp = list(pool.map(trunc_func, comp_list))
        #print(f"map time:{time()-st}")

        st=time()
        for k in range(dist.gm.n_comp()):
            if trunc_type == '==' and dist.gm.sigma[k][trunc_idx,trunc_idx] < delta_tol and sum(trans_comp[k].pi) > prob_tol:
                hard.append(k)
        #print(f"second loop time:{time()-st}")

        if len(hard) == 0:
            for k in range(dist.gm.n_comp()):
                new_mix = trans_comp[k]
                for h in range(new_mix.n_comp()):
                    if new_mix.pi[h] > prob_tol:
                        new_dist.gm.mu.append(new_mix.mu[h])
                        new_dist.gm.sigma.append(new_mix.sigma[h])
                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        else:
            for k in hard:
                new_mix = trans_comp[k]
                for h in range(new_mix.n_comp()):
                    if new_mix.pi[h] > prob_tol:
                        new_dist.gm.mu.append(new_mix.mu[h])
                        new_dist.gm.sigma.append(new_mix.sigma[h])
                        new_pi.append(dist.gm.pi[k]*new_mix.pi[h])
        norm_factor = sum(np.array(new_pi))
        if norm_factor > prob_tol:
            new_dist.gm.pi = list(np.array(new_pi)/norm_factor)
        #print(f"total time:{time()-gst}")
        return norm_factor, new_dist
    
    
def trunc_parse(var_list, trunc, data):
    """ Parses trunc using ANTLR4. Returns a function """
    lexer = TRUNCLexer(InputStream(trunc))
    stream = CommonTokenStream(lexer)
    parser = TRUNCParser(stream)
    tree = parser.trunc()
    trunc_rule = TruncRule(var_list, data)
    walker = ParseTreeWalker()
    walker.walk(trunc_rule, tree) 
    return trunc_rule


def find_basis(alpha):
    """
    Given alpha (vector of the truncation) returns a matrix A giving the change of variable necessary to make alpha one of the axis
    """
    alpha = np.array(alpha)
    u, s, v = np.linalg.svd([alpha])
    alpha1 = v[:,1:]
    A = np.vstack((alpha.reshape(1,alpha.shape[0]), alpha1.transpose()))
    return A


#def select_indices(alpha, sigma):
#    """
#    Finds the indices of the components that needs to be transformed based on the vector representation of the truncation (alpha)
#    and the covariance matrix (sigma)
#    """
#    
#    def enlarge_set(index_set):
#        total_set = index_set
#        for i in index_set:
#            i_indices = list(np.where(sigma[i,:] != 0)[0])
#            total_set = list(set(total_set + i_indices))
#        return total_set
#    
#    init_set = list(np.where(np.array(alpha)!=0)[0])
#    new_set = enlarge_set(init_set)
#    while set(init_set) != set(new_set):
#        init_set = new_set
#        new_set = enlarge_set(init_set)   
#       
#    return np.sort(new_set)  

def select_indices(alpha, aux_cov):
    alpha = np.array(alpha)
    no_zeros = np.where(alpha != 0)[0]
    coeff_set = list(copy(no_zeros))
    for idx in no_zeros:
        new_set = np.where(aux_cov[idx,:] != 0)[0]
        coeff_set += list(new_set)
    
    coeff_set = np.sort(list(set(coeff_set)))
    return coeff_set


def reduce_indices(vec, indices):
    """
    Extracts subvector/submatrix indexed by indices
    """
    try:
        vec = np.array(vec)
    except np.ComplexWarning:
        print(vec)    
    if len(vec.shape) == 1:
        red_vec = vec[indices]
    if len(vec.shape) == 2:
        red_vec = vec[indices][:,indices]
    return red_vec


def extend_indices(red_vec, old_vec, indices):
    """
    puts red_vec in the indices of old_vec
    """
    red_vec = np.array(red_vec)
    old_vec = np.array(old_vec)
    if len(old_vec.shape) == 1:
        red_vec = red_vec.reshape(len(red_vec),)
        old_vec[indices] = red_vec
    if len(old_vec.shape) == 2:
        C = old_vec[indices]
        C[:,indices] = red_vec
        old_vec[indices] = C
    return old_vec


### compute moments functions

def partitionfunc(n,k,l=0):
    """
    n is the integer to partition, k is the length of partitions, l is the min partition element size
    """
    if k < 1:
        return
    if k == 1:
        if n >= l:
            yield (n,)
        return
    for i in range(l,n+1):
        for result in partitionfunc(n-i,k-1):
            yield (i,)+result

def _prob(mu, sigma, a, b):
    """
    Computes the mass probability of the normal distribution with mean mu and covariance matrix sigma in the 
    hyper-rectangle [a,b].
    Even for one-dimensional distributions, mu, sigma, a, b must be vectors.
    """
    #n = len(mu)
    #P = 0
    #for i_list in product(*[[0,1]]*n):
    #    x = np.zeros(n)
    #    for i, idx in enumerate(i_list):
    #        if idx==0:
    #            x[i] = a[i]
    #        else:
    #            x[i] = b[i]
    #    try:
    #        p = mvnorm.cdf(x,mean=mu,cov=sigma,allow_singular=True)
    #    except ValueError:
    #        sigma = make_psd(sigma)
    #        p = mvnorm.cdf(x,mean=mu,cov=sigma,allow_singular=True)
    #    if np.isnan(p):
    #        # due to a bug in scipy (https://github.com/scipy/scipy/issues/7669), when applied to two dimensional vectors mvnorm.cdf can return nan. The problem is solvable by adding a third variable, indipendent from the others (does not affect the computed probability).
    #        new_x = list(x) + [0]
    #        new_mu = list(mu) + [0]
    #        new_sigma = list(sigma)
    #        for i in range(len(sigma)):
    #            new_sigma[i] = list(sigma[i]) + [0]
    #        new_sigma.append([0]*(len(sigma)+1))
    #        p = mvnorm.cdf(new_x, mean=new_mu, cov=new_sigma, allow_singular=True)
    #    P = P + ((-1)**(n-sum(i_list)))*p
    P = mvnorm.cdf(b[0], mean=mu[0], cov=sigma[0,0], allow_singular=True) - mvnorm.cdf(a[0], mean=mu[0], cov=sigma[0,0], allow_singular=True) 
    return P
    

#def compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc):
#    """
#    Given a normal with mean mu and cov matrix sigma,  truncated to [a,b] (where a[i] = -inf and b[i] = inf except
#    for a[trunc_idx] (if trunc = low) or b[trunc_idx] (if trunc=up)), computes the first two orders moments of a 
#    (n-1) dimensional normal distribution with mean \tilde(mu), \tilde(sigma) (as defined in Kan-Robotti).
#    """
#    n = len(mu)
#    c = np.delete(a, trunc_idx)
#    d = np.delete(b, trunc_idx)
#    # computes the new mean
#    if trunc == 'low':
#        muj = np.delete(mu, trunc_idx) + ((a[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
#    elif trunc == 'up':
#        muj = np.delete(mu, trunc_idx) + ((b[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
#    # computes the new covariance matrix
#    sigmaj = np.delete(np.delete(sigma, trunc_idx, axis=0), trunc_idx, axis=1)
#    sigmaj = sigmaj - (1/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx].reshape(len(sigma)-1,1) @         np.delete(sigma, trunc_idx, axis=1)[trunc_idx,:].reshape(1,len(sigma)-1)  
#    # saves the moments in a dictionary
#    dict_mom_lower = {}
#    for k in range(3):
#        for part in partitionfunc(k, n-1):
#            if sum(part) == 0:
#                dict_mom_lower[part] = 1
#            if sum(part) == 1:
#                idx = np.where(np.array(part) == 1)[0][0]
#                dict_mom_lower[part] = muj[idx]
#            if sum(part) == 2:
#                idx_list = np.where(np.array(part)!=0)[0]
#                if len(idx_list) == 2:
#                    idx1, idx2 = idx_list
#                    dict_mom_lower[part] = sigmaj[idx1, idx2] + muj[idx1]*muj[idx2]
#                elif len(idx_list) == 1:
#                    idx = idx_list[0]
#                    dict_mom_lower[part] = sigmaj[idx, idx] + muj[idx]**2
#    return dict_mom_lower

def compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc):
    """
    Given a normal with mean mu and cov matrix sigma,  truncated to [a,b] (where a[i] = -inf and b[i] = inf except
    for a[trunc_idx] (if trunc = low) or b[trunc_idx] (if trunc=up)), computes the first two orders moments of a 
    (n-1) dimensional normal distribution with mean \tilde(mu), \tilde(sigma) (as defined in Kan-Robotti).
    """
    n = len(mu)
    c = np.delete(a, trunc_idx)
    d = np.delete(b, trunc_idx)
    # computes the new mean
    if trunc == 'low':
        muj = np.delete(mu, trunc_idx) + ((a[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
    elif trunc == 'up':
        muj = np.delete(mu, trunc_idx) + ((b[trunc_idx]-mu[trunc_idx])/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx]
    # computes the new covariance matrix
    #sigmaj = np.delete(np.delete(sigma, trunc_idx, axis=0), trunc_idx, axis=1)
    #sigmaj = sigmaj - (1/sigma[trunc_idx, trunc_idx])*np.delete(sigma, trunc_idx, axis=0)[:,trunc_idx].reshape(len(sigma)-1,1) @         np.delete(sigma, trunc_idx, axis=1)[trunc_idx,:].reshape(1,len(sigma)-1)  
    # saves the moments in a dictionary
    #sigmaj = sigmaj + muj.reshape(-1,1).dot(muj.reshape(1,-1))
    return muj


#def _compute_mom1(n, k, mu, sigma, a, b, trunc_idx, trunc, dict_mom):
#    c = np.zeros(n)
#    idx = np.where(np.array(k)==1)[0][0]
#    if trunc == 'low':
#        c[trunc_idx] = norm.pdf(a[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
#    elif trunc == 'up':
#        c[trunc_idx] = -norm.pdf(b[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
#    return mu[idx]*dict_mom[tuple(n*[0])] + np.array(k).dot(sigma).dot(c)           

def compute_mom1(n, mu, sigma, a, b, trunc_idx, trunc, P):
    c = np.zeros(n)
    if trunc == 'low':
        c[trunc_idx] = norm.pdf(a[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
    elif trunc == 'up':
        c[trunc_idx] = -norm.pdf(b[trunc_idx], loc=mu[trunc_idx], scale=np.sqrt(sigma[trunc_idx,trunc_idx]))
    return mu + sigma.dot(c)/P     


#def _compute_mom2(n, k, mu, sigma, a, b, trunc_idx, trunc, dict_mom, dict_mom_lower):
#    c = np.zeros(n)
#    index_list = np.where(np.array(k)!=0)[0]
#    if len(index_list) == 2:
#        idxk, idxe = index_list
#        ek = np.zeros(n)
#        ek[idxk] = 1
#        e = np.zeros(n)
#        e[idxe] = 1
#        for i in range(n):
#            if i == idxk:
#                c[i] = dict_mom[tuple(n*[0])]
#            if i == trunc_idx:
#                if trunc == 'low':
#                    c[i] = c[i] + (a[i]**ek[i])*norm.pdf(a[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(ek,i))]
#                elif trunc == 'up':
#                    c[i] = c[i] - (b[i]**ek[i])*norm.pdf(b[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(ek,i))]
#        return mu[idxe]*dict_mom[tuple(ek)] + e.dot(sigma).dot(c)   
#    elif len(index_list) == 1:
#        idx = index_list[0]
#        e = np.zeros(n)
#        e[idx] = 1
#        for i in range(n):
#            if i == idx:
#                c[i] = dict_mom[tuple(n*[0])]
#            if i == trunc_idx:
#                if trunc == 'low':
#                    c[i] = c[i] + (a[i]**e[i])*norm.pdf(a[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(e,i))]
#                elif trunc == 'up':
#                    c[i] = c[i] - (b[i]**e[i])*norm.pdf(b[i], loc=mu[i], scale=np.sqrt(sigma[i,i]))*dict_mom_lower[tuple(np.delete(e,i))]
#        return mu[idx]*dict_mom[tuple(e)] + e.dot(sigma).dot(c) 
    
def compute_mom2(n, mu, sigma, a, b, trunc_idx, trunc, new_P, new_mu, muj):
    e0 = np.zeros(n)
    e0[0] = 1
    C = new_P*np.eye(n)
    if trunc == 'low':
        C[:,0] += norm.pdf(a[0], mu[0], np.sqrt(sigma[0,0]))*(a[0]**e0)*np.array([1]+list(muj))
    elif trunc == 'up':
        C[:,0] += -norm.pdf(b[0], mu[0], np.sqrt(sigma[0,0]))*(b[0]**e0)*np.array([1]+list(muj))
    new_sigma = new_P*mu.reshape(-1,1).dot(new_mu.reshape(1,-1)) + sigma.dot(np.transpose(C))
    new_sigma = new_sigma/new_P - new_mu.reshape(-1,1).dot(new_mu.reshape(1,-1))
    return new_sigma  
    

def compute_moments(mu, sigma, a, b):
    """
    Given a normal distribution with mean mu and covariance matrix sigma, truncated to [a,b], where all a_i=-np.inf and
    all b_i=np.inf except at most one a_i or one b_i, computes exactly the mean and the covariance matrix of the 
    truncated distribution
    """        
    a = np.array(a)
    b = np.array(b)
    n = len(a)   
    # truncation in one dimension
    if n==1:
        new_P = norm.cdf(b[0], loc=mu[0], scale=np.sqrt(sigma[0,0])) - norm.cdf(a[0], loc=mu[0], scale=np.sqrt(sigma[0,0]))
        new_mu, new_sigma = truncnorm.stats(loc=mu[0], scale=np.sqrt(sigma[0,0]), a=(a[0]-mu[0])/np.sqrt(sigma[0,0]), b=(b[0]-mu[0])/np.sqrt(sigma[0,0]), moments='mv')
        new_mu = np.array([new_mu])
        new_sigma = np.array([[new_sigma]])
        return new_P, new_mu, new_sigma
    # if in more dimensions applies Kan-Robotti formulas
    # first determines if the truncation is 'low' (i.e. x > c) or 'up' (i.e. x < c)
    trunc_idx = 0
    if a[0] > -np.inf:
        trunc = 'low'
    else:
        trunc = 'up'  
    # returns the moments for the distribution of dimension n-1, in which the trunc_idx component has been removed
    #dict_mom_lower = compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc)  
    muj = compute_lower_mom(mu, sigma, a, b, trunc_idx, trunc)
    # computes first two order moments using the recurrence formulas of Kan-Robotti and stores them in a dictionary
    new_P = _prob(mu, sigma, a, b)
    new_mu = compute_mom1(n, mu, sigma, a, b, trunc_idx, trunc, new_P)
    new_sigma = compute_mom2(n, mu, sigma, a, b, trunc_idx, trunc, new_P, new_mu, muj)
    #dict_mom = {}
    #for k in range(3):
    #    for part in partitionfunc(k, n): 
    #        if sum(part) == 0:
    #            dict_mom[part] = _prob(mu, sigma, a, b)
    #            if dict_mom[part] < prob_tol:
    #                return 0, mu, sigma
    #        if sum(part) == 1:
    #            dict_mom[part] = _compute_mom1(n, part, mu, sigma, a, b, trunc_idx, trunc, dict_mom)
    #        if sum(part) == 2:
    #            dict_mom[part] = _compute_mom2(n, part, mu, sigma, a, b, trunc_idx, trunc, dict_mom, dict_mom_lower)   
    ## assembles the dictionaries result in new_P, new_mu, new_sigma
    #new_P = dict_mom[tuple(n*[0])]
    #new_mu = np.zeros(n)
    #new_sigma = np.zeros((n,n))
    #for i in range(n):
    #    e = np.zeros(n)
    #    e[i] = 1
    #    new_mu[i] = dict_mom[tuple(e)]/new_P
    #    new_sigma[i,i] = dict_mom[tuple(2*e)]/new_P - (dict_mom[tuple(e)]/new_P)**2
    #    for j in range(i):
    #        f = np.zeros(n)
    #        f[j] = 1
    #        new_sigma[i,j] = new_sigma[j,i] = dict_mom[tuple(e+f)]/new_P - (dict_mom[tuple(e)]/new_P)*(dict_mom[tuple(f)]/new_P)
    return new_P, new_mu, new_sigma

### conditioning to zero probability events

def insert_value(val, idx, mu, sigma):
    """ Extends mu and sigma by adding val in corresponding to the idx position (for sigma the other row- and column-entries are 0) """
    d = len(mu)
    new_mu = np.array(list(mu[:idx]) + [val] + list(mu[idx:]))
    new_sigma = np.block([[sigma[:idx,:idx], np.zeros((idx,1)), sigma[:idx,idx:]], 
          [np.zeros((1,d+1))],
          [sigma[idx:,:idx], np.zeros((d-idx,1)), sigma[idx:,idx:]]])
    return new_mu, new_sigma
    
    
    