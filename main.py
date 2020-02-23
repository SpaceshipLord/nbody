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
field = np.zeros((globals.n, 3))

field = direct_method.main(globals.n, globals.pos, globals.mass, globals.G, globals.eps)

# evolving nt time steps
for i in range(globals.nt):
    
    # writing positions
    np.save('./results/pos_'+str(i)+'.npy', globals.pos)
    
    # evaluating gravitational force
    field = direct_method.main(globals.n, globals.pos, globals.mass, globals.G, globals.eps)
    
    # evolving position and velocity
    pos_vel = leap_frog.main(globals.n, globals.pos, globals.vel, globals.mass, field, globals.dt)
    
    globals.pos, globals.vel = pos_vel[0], pos_vel[1]
    
t2 = time.time()
print(t2-t1)





