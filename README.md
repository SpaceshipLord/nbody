# nbody

A tool to simulate the a dynamical system of particles under the influence of gravitation. At the moment, the gravitational field is evaluated using a direct particle-to-particle approach and the system is evolved in time using the leap frog integration. The tool is written in python and a large part of the project is to improve the performance of the written code as much as possible. 

Stuff, I am looking forward to look into and include is: different methods to evaluate the gravitational field (tree-code, FMM, etc.), different integration schemes (Runge-Kutta, adaptive methods), the usage of Cython and parallel computing. 