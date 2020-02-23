from numba import jit
import numpy as np

@jit
def main(n, pos, mass, G, eps): 
    
    """
    This function evaluates the gravitational field at the positions of all 
    particles (i) in the vectorial form of Newton's law of universal gravitation
    using a direct particle-to-particle approach. 
    
    credits
    =======
    https://mikegrudic.wordpress.com/2017/07/11/a-simple-and-pythonic-barnes-hut-treecode/
    """
    
    # data dictionary
    # ===============
    
    field = np.zeros((n, 3))    # gravitational field
    field_tmp = 0.              # temporial gravitational field to save
    dist_vector = np.zeros(3)   # vectorial distance between particles
    dist_scalar = 0.            # scalar distance between particles
    
    # evaluating gravitational forces
    # ===============================
    
    # using the fact, that F(j->i) = -F(i->j)
    for i in range(n):
        for j in range(i+1, n):    
            
            # evaluating distance between particle i and particle j
            for k in range(3):
                dist_vector[k] = pos[i,k] - pos[j,k]
                dist_scalar += dist_vector[k]**2            
            dist_scalar = (dist_scalar+eps**2)**1.5   
            
            # evaluating gravitational field on position of particle i acting from particle j
            for k in range(3):
                field_tmp = -G*mass[j]/dist_scalar*dist_vector[k]
                field[i,k] += field_tmp
                field[j,k] -= field_tmp/mass[j]*mass[i]    
                
            dist_scalar = 0.
            
    return(field)
            
            
            
if __name__ == '__main__':
    
    import timeit
    
    # no particle
    n = 0
    pos = np.random.uniform(0, 1, (n, 3))
    mass = np.ones(n)
    G = 1.
    eps = 0.
    print(main(n, pos, mass, G, eps))
    
    # one partile
    n = 1
    pos = np.random.uniform(0, 1, (n, 3))
    mass = np.ones(n)
    G = 1.
    eps = 0.
    print(main(n, pos, mass, G, eps))
    
    # two particles in such a way, that:
    # F(0) = (-0.1925, -0.1925, -0.1925)
    # F(1) = ( 0.1925,  0.1925,  0.1925)
    n = 2
    pos = np.array([[1, 1, 1], [0, 0, 0]])
    mass = np.ones(n)
    G = 1. 
    eps = 0.
    print(main(n, pos, mass, G, eps))
    
    # four particles in such a way, that:
    # F(0) = (-1.0424, -0.4062,  0.4062)
    # F(1) = (-0.6139,  0.5685, -0.8527)
    # F(2) = ( 0.4743,  0.4629,  0.9258)
    # F(3) = ( 0.4058, -0.5489,  0.5489)
    res = np.zeros(3)
    n = 4
    pos = np.array([[2,2,0], [2,0,2], [0,1,1], [0,2,0]])
    mass = np.array([1., 1.5, 2., 2.5])
    G = 3.
    eps = 2.
    print(main(n, pos, mass, G, eps))

    # performance test
    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped  
    n = int(1e+3)
    pos = np.random.uniform(0, 1000, (n,3))
    mass = np.ones(n)
    G = 9.81
    eps = 2.
    wrapped = wrapper(main, n, pos, mass, G, eps)
    print(timeit.timeit(wrapped, number = 10)/10) # 4e-2s for 1e+3 particles