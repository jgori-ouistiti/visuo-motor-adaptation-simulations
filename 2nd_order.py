import numpy
import matplotlib.pyplot as plt
import lqg_core as lqg

# ================== Intro ===========
# Simulation based on a steady state LQG formulation by Philis 1985, with a classical spring-masss-damper system
# Equations:
# X = X + (A@X + B*U[i,nt])*timestep + F@X*beta + Y*U[i,nt]*gamma + G@omega
# dy =  C@X*timestep + D@ksi
# Xhat = Xhat + (A@Xhat + B*U[i,nt])*timestep + K@(dy - C@Xhat*timestep)


m = 1
d = 1.2
k = 3
x0 = numpy.array([[0.2],[0]])

A = numpy.array([   [0, 1],
                    [-k/m, -d/m],
                    ])

B = numpy.array([[ 0, 1/m]]).reshape((-1,1))

C = numpy.eye(2)

D = numpy.array([   [0.1, 0],
                    [0, 1]
                        ])

F = 1*numpy.diag([0.01, 0.01])
Y = 0.06*B
G = 0.005*numpy.diag([1,0.1])


Q = numpy.diag([1, 0])
R = numpy.array([[1e-4]])
U = numpy.diag([1, 0.1])


D = D*0.35
G = G*0.35



# Initialize simulation parameters

timestep =  2e-2
TF = 5
ntrials = 25

rho = .9

Ac, Bc, Cc = A, B, C
# Ac = A + numpy.array([[0,0],[(numpy.random.random()-0.5)*rho, (numpy.random.random()-0.5)*rho]])
# Bc = B + numpy.array([[0],[(numpy.random.random()-0.5)*rho]])
# Cc = C + numpy.diag([(numpy.random.random()-.5)*rho, (numpy.random.random()-.5)*rho])


ret = lqg.compute_kalman_matrices(Ac, Bc, Cc, D, F, G, Q, R, U, Y)
K,L = ret




X0 = [-0.5]
for u in range(A.shape[0]-1):
    X0.append(0)
init_value = [X0, X0]

Mov, U = lqg.simulate_N_movements(timestep, TF, A, B, C, D, Ac, Bc, Cc, F, G, K, L, Y, noise = 'on', ntrials = ntrials, init_value = init_value)


time = [-timestep] + numpy.arange(0,TF,timestep).tolist()
fig = plt.figure()
ax = fig.add_subplot(131)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (m)')
axhat = fig.add_subplot(132)
axhat.set_xlabel('Time (s)')
axhat.set_ylabel('Estimated Position (m)')
au = fig.add_subplot(133)
au.set_xlabel('Time (s)')
au.set_ylabel('Control')
for nt in range(ntrials):
    ax.plot(time, Mov[:,nt,0,0], '-')
    axhat.plot(time, Mov[:,nt,0,1], '-')
    au.plot(time[1:], U[:,nt], '-')



plt.tight_layout()
plt.show()
