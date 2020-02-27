import globals
import direct_method
import leap_frog
import time
import numpy as np

# nbody simulation
# ================

t1 = time.time()

# initial calculation
globals.main()

pos_vel = np.zeros((2, globals.n, 3))

# evolving nt time steps
for i in range(globals.nt):
    
    # writing positions
    np.save('./results/pos_'+str(i)+'.npy', globals.pos)
    
    # evolving position and velocity
    pos_vel = leap_frog.main(globals.n, globals.pos, globals.vel, globals.dt, globals.mass, globals.G, globals.eps)
    
    globals.pos, globals.vel = pos_vel[0], pos_vel[1]
    
t2 = time.time()
print(t2-t1)





