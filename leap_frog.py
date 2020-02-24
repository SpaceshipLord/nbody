from numba import jit
import numpy as np

@jit
def main(n, pos, vel, field, dt):
    
    """
    This function evolves the positions and the velocities of the particles for 
    a single time step using the leap frog integration.
    """
    
    # data dictionary
    # ===============
    
    res = np.zeros((2, n, 3))   # results [pos, vel]
    
    # time evolution
    # ==============
    
    # updating positions
    for i in range(n):
        for k in range(3):
            res[0,i,k] = pos[i,k]+vel[i,k]*dt
    
    # updating velocities
    for i in range(n):
        for k in range(3):
            res[1,i,k] = vel[i,k]+field[i,k]*dt
            
    return(res)
            
            
            
if __name__ == '__main__':
    
    import timeit
    import direct_method
    
    # no particle
    n = 0
    pos = np.random.uniform(0, 1, (n, 3))
    vel = np.random.uniform(0, 1, (n, 3))
    mass = np.random.uniform(0, 1, n)
    dt = 1.
    G = 1.
    eps = 0.
    field = direct_method.main(n, pos, mass, G, eps)
    res = main(n, pos, vel, field, dt)
    print(np.all(pos == res[0]))
    
    # one particle
    n = 1
    pos = np.random.uniform(0, 1, (n, 3))
    vel = np.random.uniform(0, 1, (n, 3))
    mass = np.random.uniform(0, 1, n)
    dt = 1.
    G = 1.
    eps = 0.
    field = direct_method.main(n, pos, mass, G, eps)
    res = main(n, pos, vel, field, dt)
    print(np.all((pos+vel*dt) == res[0]))
    
    # two particles
    n = 2
    pos = np.random.uniform(0, 1, (n, 3))
    vel = np.random.uniform(0, 1, (n, 3))
    mass = np.random.uniform(0, 1, n)
    dt = 1.
    G = 1.
    eps = 0.
    field = direct_method.main(n, pos, mass, G, eps)
    res = main(n, pos, vel, field, dt)
    print(np.all((pos+vel*dt) == res[0]))
    print(np.all((vel+field*dt) == res[1]))
    
    # performance test
    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped   
    n = int(1e+3)
    pos = np.random.uniform(0, 1000, (n,3))
    vel = np.random.uniform(0, 1000, (n,3))
    mass = np.ones(n)
    dt = 1.
    G = 9.81
    eps = 2.
    field = direct_method.main(n, pos, mass, G, eps)
    wrapped = wrapper(main, n, pos, vel, field, dt)
    print(timeit.timeit(wrapped, number = 10)/10) # 1e-5s for 1e+3 particles
