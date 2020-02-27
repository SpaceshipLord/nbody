from numba import jit
import numpy as np
import direct_method

@jit
def main(n, pos, vel, dt, mass, G, eps):
    
    """
    This function evolves the positions and velocities of the particles using 
    the leap frog integration:
        
        v(i+1/2) = v(i)     + a(i)*dt/2     kick
        x(i+1)   = x(i)     + v(i+1/2)*dt   drift
        v(i+1)   = v(i+1/2) + a(i+1)*dt/2   kick
    """
    
    # data dictionary
    # ===============
    
    field = np.zeros((n,3))
    res = np.zeros((2,n,3)) # [pos, vel]
    
    # time evolution
    # ==============
    
    field = direct_method.main(n, pos, mass, G, eps)
    
    for i in range(n):
        for j in range(3):
            res[1,i,j] = vel[i,j] + field[i,j]*dt/2 # kick
            res[0,i,j] = pos[i,j] + res[1,i,j]*dt # drift
            
    field = direct_method.main(n, res[0,:,:], mass, G, eps)
    
    for i in range(n):
        for j in range(3):
            res[1,i,j] += field[i,j]*dt/2 # kick
            
    return(res)

    

            
if __name__ == '__main__':
    
    import timeit
    
    # no particle
    n = 0
    pos = np.random.uniform(0, 1, (n, 3))
    vel = np.random.uniform(0, 1, (n, 3))
    mass = np.random.uniform(0, 1, n)
    dt = 1.
    G = 1.
    eps = 0.
    res = main(n, pos, vel, dt, mass, G, eps)
    print(np.all(pos == res[0]))
    
    # one particle
    n = 1
    pos = np.random.uniform(0, 1, (n, 3))
    vel = np.random.uniform(0, 1, (n, 3))
    mass = np.random.uniform(0, 1, n)
    dt = 1.
    G = 1.
    eps = 0.
    res = main(n, pos, vel, dt, mass, G, eps)
    print(np.all((pos+vel*dt) == res[0]))
    
    # two particles
    n = 2
    pos = np.random.uniform(0, 1, (n, 3))
    vel = np.random.uniform(0, 1, (n, 3))
    mass = np.random.uniform(0, 1, n)
    dt = 1.
    G = 1.
    eps = 0.
    res = main(n, pos, vel, dt, mass, G, eps)
    field = direct_method.main(n, pos, mass, G, eps)
    vel = vel+field*dt/2
    pos = pos+vel*dt
    field = direct_method.main(n, pos, mass, G, eps)
    vel = vel+field*dt/2
    print(np.all(pos == res[0]))
    print(np.all(vel == res[1]))
    
    
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
    wrapped = wrapper(main, n, pos, vel, dt, mass, G, eps)
    print(timeit.timeit(wrapped, number = 1000)/1000) # 5e-2s for 1e+3 particles
