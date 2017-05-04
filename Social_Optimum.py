__author__ = "Jerome Thai"
__email__ = "jerome.thai@berkeley.edu"


'''
This module is frank-wolfe algorithm using an all-or-nothing assignment
based on igraph package
'''

import numpy as np
from process_data import construct_igraph, construct_od
from AoN_igraph import all_or_nothing
#Profiling the code
import timeit

#This finds the potential for social optimum case
def potential_Social_Optimum(graph ,f):
    #Finds the potential in the case of social optimum
    links = int(np.max(graph[:,0])+1)
    #The potential in social optimum is a0*f+a1*f^2+a2*f^3+a3*f^4+a4*a^5
    x = np.power(f.reshape((links,1)), np.array([1,2,3,4,5]))
    return np.sum(np.einsum('ij,ij->i', x, graph[:,3:]))


def line_search(f, res=20):
    # on a grid of 2^res points bw 0 and 1, find global minimum
    # of continuous convex function
    d = 1./(2**res-1)
    l, r = 0, 2**res-1
    while r-l > 1:
        if f(l*d) <= f(l*d+d): return l*d
        if f(r*d-d) >= f(r*d): return r*d
        # otherwise f(l) > f(l+d) and f(r-d) < f(r)
        m1, m2 = (l+r)/2, 1+(l+r)/2
        if f(m1*d) < f(m2*d): r = m1
        if f(m1*d) > f(m2*d): l = m2
        if f(m1*d) == f(m2*d): return m1*d
    return l*d


def total_free_flow_cost(g, od):
    return np.array(g.es["weight"]).dot(all_or_nothing(g, od))

#Calculates the total travel cost/time
def total_cost(graph, f, grad):
    #Since the cost function equals to t(f) = a0+a1*f+a2*f^2+a3*f^3+a4*f^4 (where f is the flow)
    #the travel cost, f*t(f) = a0*f+ a1*f^2 + a2*f^3+ a3*f^4 + a4*f^5
    x = np.power(f.reshape((f.shape[0],1)), np.array([1,2,3,4,5]))  # x is a matrix containing f,f^2, f^3, f^4, f^5
    tCost = np.sum(np.einsum('ij,ij->i', x, graph[:,3:]))    # Multiply matrix x with coefficients a0, a1, a2, a3 and a4
    return tCost
    


#This function allow each user to account for their impact on the global travel cost
#thus allowing to find the social optimum
def search_direction_social_optimum(f, graph, g, od):
    # computes the Frank-Wolfe step
    # g is just a canvas containing the link information and to be updated with 
    # the most recent edge costs
    x = np.power(f.reshape((f.shape[0],1)), np.array([0,1,2,3,4]))
    #import pdb; pdb.set_trace()
    #When we add the price of anarchy, the cost function becomes a0+ 2*a2*x+ 3*a3*x^2+ 4*a4*x^3+ 5*a5*x^4
    #Matrix coefficients saves the 1, 2, 3, 4, and 5 coefficients in front of the ai*x terms
    coefficients = np.array([1,2,3,4,5])
    onesArray = np.ones((f.shape[0],5))
    coefficients = onesArray * coefficients.transpose()
    #y stands for the flow multiplied by the coefficients
    #import pdb; pdb.set_trace()
    y = np.einsum('ij, ij->ij', x, coefficients) 
    grad = np.einsum('ij,ij->i', y, graph[:,3:])
    g.es["weight"] = grad.tolist()
  
    #Calculating All_or_nothing
    L = all_or_nothing(g, od)

    return L, grad


def solver_social_optimum(graph, demand, g=None, od=None, past=10, max_iter=100, eps=1e-16, \
    q=50, display=0, stop=1e-8):
    '''
    this is an adaptation of Fukushima's algorithm
    graph:    numpy array of the format [[link_id from to a0 a1 a2 a3 a4]]
    demand:   mumpy arrau of the format [[o d flow]]
    g:        igraph object constructed from graph
    od:       od in the format {from: ([to], [rate])}
    past:     search direction is the mean over the last 'past' directions
    max_iter: maximum number of iterations
    esp:      used as a stopping criterium if some quantities are too close to 0
    q:        first 'q' iterations uses open loop step sizes 2/(i+2)
    display:  controls the display of information in the terminal
    stop:     stops the algorithm if the error is less than 'stop'
    '''

    assert past <= q, "'q' must be bigger or equal to 'past'"
    if g is None: 
        g = construct_igraph(graph)
    if od is None: 
        od = construct_od(demand)
    f = np.zeros(graph.shape[0],dtype="float64") # initial flow assignment is null
    fs = np.zeros((graph.shape[0],past),dtype="float64") #not sure what fs does
    K = total_free_flow_cost(g, od)

    # why this?
    if K < eps:
        K = np.sum(demand[:,2])
    elif display >= 1:
        print 'average free-flow travel time', K / np.sum(demand[:,2])

    for i in range(max_iter):

        if display >= 1:
            if i <= 1:
                print 'iteration: {}'.format(i+1)
            else:            
                print 'iteration: {}, error: {}'.format(i+1, error)

        # search direction using social optimum
        L, grad = search_direction_social_optimum(f, graph, g, od)


        fs[:,i%past] = L
        w = L - f
        if i >= 1:
            error = -grad.dot(w) / K
            # if error < stop and error > 0.0:
            if error < stop:
                if display >= 1: print 'stop with error: {}'.format(error)
                #printing the final total cost
                average_cost = total_cost(graph, L, grad)/np.sum(demand[:,2])
                print ("average delay %s seconds" % average_cost)
                #import pdb; pdb.set_trace()

                return f
        if i > q:
            # step 3 of Fukushima
            v = np.sum(fs,axis=1) / min(past,i+1) - f
            norm_v = np.linalg.norm(v,1)
            if norm_v < eps: 
                if display >= 1: print 'stop with norm_v: {}'.format(norm_v)
                #printing the final total cost
                average_cost = total_cost(graph, L, grad)/np.sum(demand[:,2])

                print ("average delay %s seconds" % average_cost)
                return f
            norm_w = np.linalg.norm(w,1)
            if norm_w < eps: 
                if display >= 1: print 'stop with norm_w: {}'.format(norm_w)
                #printing the final total cost
                average_cost = total_cost(graph, L, grad)/np.sum(demand[:,2])
                print ("average delay %s seconds" % average_cost)

                return f
            # step 4 of Fukushima
            gamma_1 = grad.dot(v) / norm_v
            gamma_2 = grad.dot(w) / norm_w
            if gamma_2 > -eps: 
                if display >= 1: print 'stop with gamma_2: {}'.format(gamma_2)
                average_cost = total_cost(graph, L, grad)/np.sum(demand[:,2])
                print ("average delay %s seconds" % average_cost)
                return f
            d = v if gamma_1 < gamma_2 else w
            # step 5 of Fukushima
            s = line_search(lambda a: potential_Social_Optimum(graph, f+a*d))
            if s < eps: 
                if display >= 1: print 'stop with step_size: {}'.format(s)
                #printing the final total cost
                average_cost = total_cost(graph, L, grad)/np.sum(demand[:,2])
                print ("average delay %s seconds" % average_cost)
                #import pdb; pdb.set_trace()
                return f
            f = f + s*d
        else:
            f = f + 2. * w/(i+2.)  

    return f

