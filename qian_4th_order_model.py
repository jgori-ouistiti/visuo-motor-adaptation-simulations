import numpy
import matplotlib.pyplot as plt
import lqg_core as lqg

# ================== Intro ===========
# Simulation based on a steady state LQG formulation by Philis 1985, with values used in Qian 2013
# Equations:
# X = X + (A@X + B*U[i,nt])*timestep + F@X*beta + Y*U[i,nt]*gamma + G@omega
# dy =  C@X*timestep + D@ksi
# Xhat = Xhat + (A@Xhat + B*U[i,nt])*timestep + K@(dy - C@Xhat*timestep)

# Using data from Qian 2013
I = 0.25
b = 0.2
ta = 0.03
te = 0.04

a1 = b/(ta*te*I)
a2 = 1/(ta*te) + (1/ta + 1/te)*b/I
a3 = b/I + 1/ta + 1/te
bu = 1/(ta*te*I)


A = numpy.array([   [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [0, -a1, -a2, -a3]    ])

B = numpy.array([[ 0, 0, 0, bu]]).reshape((-1,1))

C = numpy.array([   [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                        ])



D = numpy.array([   [0.01, 0, 0],
                    [0, 0.01, 0],
                    [0, 0, 0.05]
                    ])

F = numpy.diag([0.01, 0.01, 0.01, 0.01])
Y = 0.08*B
G = 0.03*numpy.diag([1,0.1,0.01,0.001])


Q = numpy.diag([1, 0.01, 0, 0])
R = numpy.array([[1e-4]])
U = numpy.diag([1, 0.1, 0.01, 0])


D = D*0.35
G = G*0.35



# Initialize simulation parameters

timestep = 1e-2
TF = 5
ntrials = 25

ret = lqg.compute_kalman_matrices(A, B, C, D, F, G, Q, R, U, Y)
K,L = ret
X0 = [1,0,0,0]

for i in [0.1, 0.5, 1, 1.5, 2]:
    for j in [0.5, 1, 1.5, 2]:
        Ac, Bc, Cc = i*A, j*B, C
        Mov, U = lqg.simulate_N_movements(timestep, TF, A, B, C, D, Ac, Bc, Cc, F, G, K, L, Y, noise = 'on', ntrials = ntrials, init_value = [X0, X0])
        time = [-timestep] + numpy.arange(0,TF,timestep).tolist()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        # axhat = fig.add_subplot(132)
        # axhat.set_xlabel('Time (s)')
        # axhat.set_ylabel('Estimated Position (m)')
        # au = fig.add_subplot(133)
        # au.set_xlabel('Time (s)')
        # au.set_ylabel('Control')
        for nt in range(ntrials):
            ax.plot(time, Mov[:,nt,0,0], '-')
            # axhat.plot(time, Mov[:,nt,0,1], '-')
            # au.plot(time[1:], U[:,nt], '-')

        plt.tight_layout()
        plt.savefig("{}A_{}B.pdf".format(i,j))
        plt.close()
