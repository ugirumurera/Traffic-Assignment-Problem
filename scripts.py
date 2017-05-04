__author__ = "Jerome Thai"
__email__ = "jerome.thai@berkeley.edu"

'''
Various scripts for processing data
'''

import numpy as np
from process_data import extract_features, process_links, geojson_link, \
    process_trips, process_net, process_node, array_to_trips, process_results
from metrics import average_cost, cost_ratio, cost, save_metrics
from frank_wolfe import solver, solver_2, solver_3
from heterogeneous_solver import gauss_seidel, jacobi
from All_Or_Nothing import all_or_nothing
from scripts_LA import load_LA
import timeit


def test_anaheim(self):
    print 'test Frank-Wolfe on Anaheim'
    graph = np.loadtxt('data/Anaheim_net.csv', delimiter=',', skiprows=1)
    demand = np.loadtxt('data/Anaheim_od.csv', delimiter=',', skiprows=1)        
    demand[:,2] = demand[:,2] / 4000
    f = solver(graph, demand, max_iter=1000)
    # print f.shape
    results = np.loadtxt('data/Anaheim_results.csv')
    print np.linalg.norm(f*4000 - results) / np.linalg.norm(results)
    # print f*4000


def braess_heterogeneous(demand_r, demand_nr):
    # generate heteregenous game on Braess network
    g_r = np.loadtxt('data/braess_net.csv', delimiter=',', skiprows=1)
    g_nr = np.copy(g_r)
    g_nr[2,3] = 1e8
    d_nr = np.loadtxt('data/braess_od.csv', delimiter=',', skiprows=1)
    d_nr = d_nr.reshape((1,3))
    d_nr[0,2] = demand_nr
    d_r = np.copy(d_nr)
    d_r[0,2] = demand_r
    return g_nr, g_r, d_nr, d_r


def braess_parametric_study():
    '''
    parametric study of heterogeneous game on the Braess network 
    '''
    g1,g2,d1,d2 = braess_heterogeneous(.0, 1.5)
    fs = solver_2(g1, d1, display=1)
    print '.0, 1.5'
    np.savetxt('data/braess/test_1.csv', fs, delimiter=',')
    g1,g2,d1,d2 = braess_heterogeneous(.5, 1.)
    fs = gauss_seidel([g1,g2], [d1,d2], solver_2, display=1)
    print '.5, 1.'
    np.savetxt('data/braess/test_2.csv', fs, delimiter=',')
    g1,g2,d1,d2 = braess_heterogeneous(.75, .75)
    fs = gauss_seidel([g1,g2], [d1,d2], solver_2, display=1)
    print '.75, .75'
    np.savetxt('data/braess/test_3.csv', fs, delimiter=',')
    g1,g2,d1,d2 = braess_heterogeneous(1., .5)
    fs = gauss_seidel([g1,g2], [d1,d2], solver_2, display=1)
    print '1., .5'
    np.savetxt('data/braess/test_4.csv', fs, delimiter=',')
    g1,g2,d1,d2 = braess_heterogeneous(1.5, .0)
    fs = solver_2(g2, d2, display=1)
    print '1.5, .0'
    np.savetxt('data/braess/test_5.csv', fs, delimiter=',')

#Added a new function to test all_or_nothign cython code
def Cython_Func_LA():
        #graph = np.loadtxt('data/SiouxFalls_net.csv', delimiter=',', skiprows=1)
        #demand = np.loadtxt('data/SiouxFalls_od.csv', delimiter=',', skiprows=1)
    graph = np.loadtxt('data/LA_net.csv', delimiter=',', skiprows=1)
    demand = np.loadtxt('data/LA_od_2.csv', delimiter=',', skiprows=1)
    graph[10787,-1] = graph[10787,-1] / (1.5**4)
    graph[3348,-1] = graph[3348,-1] / (1.2**4)

    demand[:,2] = 0.5*demand[:,2] / 4000
    #import pdb; pdb.set_trace()
    f = solver_3(graph, demand, max_iter=1000)
    #results = np.loadtxt('data/SiouxFalls_results.csv')
    np.savetxt('data/la/LA_Cython.csv', f, delimiter=',')
    #self.check(f*4000, results, 1e-3)

def visualize_LA():
    net, demand, node = load_LA()
    f = np.loadtxt('data/la/LA_Cython.csv', delimiter=',', skiprows=0)
    features = np.zeros((f.shape[0], 4))
    features[:,:3] = extract_features('data/LA_net.txt')
    #import pdb; pdb.set_trace()
    f = np.divide(f*4000, features[:,0])
    features[:,3] = f
    links = process_links(net, node, features, in_order=True)
    #creates color array used to visulized the links
    #values useful in differenciating links based of flow on links
    color = 2.0*f + 1.0
    #congestion = f/features[:,0]    #determines the congestion levels of links
    geojson_link(links, ['capacity', 'length', 'fftt', 'flow_over_capacity'], color)


def main():
    #braess_parametric_study()
    #start timer
    start_time2 = timeit.default_timer()
    Cython_Func_LA()
    #end of timer
    elapsed2 = timeit.default_timer() - start_time2;
    print ("Execution took %s seconds" % elapsed2)
    visualize_LA()


if __name__ == '__main__':
    main()