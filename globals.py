import numpy as np

def main():
    
    """
    This function initialises all parameters (n, d, eps, dens, G, dt, nt) and 
    variables (pos, vel, mass, rad, f) as globals in order to be used by the 
    other functions.
    
    The masses are drawn from a uniform distribution with low = 1 and high = 100.
    The density is set in such a way, that a particle with mass = 1. has a radius
    of r = 0.5. The positions are a random sphere with radius = d/2 and the 
    velocities are drawn from a uniform distribution with loc = 0 and scale = 500.
    
    credits
    =======
    http://corysimon.github.io/articles/uniformdistn-on-sphere/
    https://stats.stackexchange.com/questions/120527/simulate-a-uniform-distribution-on-a-disc
    """
    
    # data dictionary
    # ===============
    
    # parameters
    global n; n = int(1e+3)                 # number of particles
    global d; d = 1e+3                      # simulation box size
    global eps; eps = 1.                    # softening length
    global dens; dens = 1/(np.pi*.5**3)*.75 # density of all particles
    global G; G = 9.81                      # gravitational constant
    global dt; dt = .01                     # length time step
    global nt; nt = int(100)                # number of time steps                           
    
    # variables
    global pos; pos = np.zeros((n, 3))                  # init position of particles
    global vel; vel = np.random.normal(0, 500, (n, 3))  # init velocity of particles
    global mass; mass = np.random.uniform(1, 100, n)    # init mass of particles
    global rad; rad = (mass*.75/(dens*np.pi))**(1/3)    # init radius of particles
    
    # local
    r = np.zeros(n)     # radius for sampling positions
    th = np.zeros(n)    # angle theta for sampling positions
    phi = np.zeros(n)   # angle phi for sampling positions
    
    # initialisation of positions
    # ===========================
                      
    r = (np.random.uniform(0,1,n))**(1/3)
    th = np.pi*np.random.uniform(0, 2, n)
    phi = np.arccos(1-2*np.random.uniform(0, 1, n))
    pos[:, 0] = (r*np.sin(phi)*np.cos(th))*d/2
    pos[:, 1] = (r*np.sin(phi)*np.sin(th))*d/2
    pos[:, 2] = (r*np.cos(phi))*d/2
    
    
    
if __name__ == '__main__':
    
    from mpl_toolkits.mplot3d import Axes3D  
    import matplotlib.pyplot as plt
    import time    
    
    # performance test
    t1 = time.time()
    main()
    t2 = time.time()
    print(t2-t1) # 5e-4s for 1e+3 particles
    
    # to look at the initial mass distribution
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(mass, bins = 100)
    ax.set_title('initial mass distribution')
    
    # to look at the initial partile distribution  
    fig = plt.figure(figsize=(8, 8), dpi = 100)
    ax = fig.add_subplot(111, projection='3d', aspect = 'equal', facecolor = 'black')
    ax.set_xlim(-d/2, d/2)
    ax.set_ylim(-d/2, d/2)
    ax.set_zlim(-d/2, d/2)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], s = rad, c = 'white', marker = '.')
    ax.axis("off")
    
    # to look at the initial velocity distribution in x
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(vel[:,1], bins = 100)
    ax.set_title('initial velocity distribution')    