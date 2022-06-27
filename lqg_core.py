import numpy
import matplotlib.pyplot as plt
import random
import math



def LinRicatti(A, B, C):
    # Compute the (L2) norm of the equation of the form AX + XA.T + BXB.T + C = 0
    n,m = A.shape
    nc,mc = C.shape
    if n !=m:
        print('Matrix A has to be square')
        return -1
    M = numpy.kron( numpy.identity(n), A ) + numpy.kron( A, numpy.identity(n) ) + numpy.kron(B,B)
    C = C.reshape(-1, 1)
    X = -numpy.linalg.pinv(M)@C
    # X = -numpy.linalg.inv(M.T@M)@M.T@C
    X = X.reshape(n,n)
    C = C.reshape(nc,mc)
    res = numpy.linalg.norm(A@X + X@A.T + B@X@B.T + C)
    return X, res



def compute_kalman_matrices(A, B, C, D, F, G,  Q, R, U, Y,  N = 20, check = 'validator'):
    K = numpy.random.rand(*C.T.shape)
    L = numpy.random.rand(1, A.shape[1])

    Lnorm = []
    Knorm = []
    Pnorm = []
    Snorm = []

    for i in range(N):
        # print(i)
        Lnorm.append(numpy.linalg.norm(L))
        Knorm.append(numpy.linalg.norm(K))

        n,m = A.shape
        Abar = numpy.block([
            [A - B@L, B@L],
            [numpy.zeros((n,m)), A - K@C]
        ])

        Ybar = numpy.block([
            [-Y@L, Y@L],
            [-Y@L, Y@L]
        ])



        Gbar = numpy.block([
            [G, numpy.zeros((G.shape[0], D.shape[1]))],
            [G, -K@D]
        ])

        V = numpy.block([
            [Q + L.T@R@L, -L.T@R@L],
            [-L.T@R@L, L.T@R@L + U]
        ])

        P, p_res = LinRicatti(Abar, Ybar, Gbar@Gbar.T)
        S, s_res = LinRicatti(Abar.T, Ybar.T, V)

        Pnorm.append(numpy.linalg.norm(P))
        Snorm.append(numpy.linalg.norm(S))

        P22 = P[n:,n:]
        S11 = S[:n,:n]
        S22 = S[n:,n:]

        K = P22@C.T@numpy.linalg.pinv(D@D.T)
        L = numpy.linalg.pinv(R + Y.T@(S11 + S22)@Y)@B.T@S11

    K, L = check_KL(Knorm, Lnorm, A, B, C, D, F, G,  K, L, Q, R, U, Y, mode = check)
    return K,L


def counted_decorator(f):
        def wrapped(*args, **kwargs):
            wrapped.calls += 1
            return f(*args, **kwargs)
        wrapped.calls = 0
        return wrapped

@counted_decorator
def check_KL( Knorm, Lnorm, A, B, C, D, F, G,  K, L, Q, R, U, Y, mode = 'validator'):
    average_delta = numpy.convolve(numpy.diff(Lnorm) + numpy.diff(Knorm), numpy.ones(5)/5, mode='full')[-5]
    if mode == 'converger':
        if check_KL.calls == 6:
            print('warning, procedure did not converge')
            return None, None
        elif average_delta > 0.01: # Arbitrary threshold
            print('Warning: the K and L matrices computations did not converge. Retrying with different starting point and a N={:d} search'.format(int(20*1.3**check_KL.calls)))
            K, L = compute_kalman_matrices(A, B, C, D, F, G,  Q, R, U, Y, N=int(20*1.3**check_KL.calls), mode = 'converger')
        else:
            return K, L
    elif mode == 'validator':
        if average_delta > 0.01:
            return None, None
        else:
            return K, L




def simulate_N_movements(timestep, TF, A, B, C, D, Ac, Bc, Cc, F, G, K, L, Y, noise = 'on', ntrials = 25, init_value = None):
    if init_value is None:
        X0 = [numpy.random.random()-0.5]
        for u in range(A.shape[0]-1):
            X0.append(0)
        init_values = [X0, X0]
        
    else:
        init_values = init_value


    time = [-timestep] + numpy.arange(0,TF,timestep).tolist()
    Mov = numpy.zeros((len(time),ntrials, A.shape[0], 2))
    Mov[0,:,:,0] = init_values[0]
    Mov[0,:,:,1] = init_values[1]
    U = numpy.zeros((len(time)-1,ntrials))
    for i,t in enumerate(time[1:]):
        print(i)
        for nt in range(ntrials):
            Mov, U = step_movement(timestep, i, nt, Mov, U, A, B, C, D, Ac, Bc, Cc, F, G, K, L, Y, noise = noise)
    return Mov, U

def step_movement(timestep, i, nt, Mov, U, A, B, C, D, Ac, Bc, Cc, F, G, K, L, Y, noise = 'off'):
    X, Xhat = Mov[i,nt,:,0].reshape(-1,1), Mov[i,nt,:,1].reshape(-1,1)
    try:
        U[i,nt] = (-L@Xhat)[0,0]
    except TypeError:
        U[i,nt] = -L@Xhat
    if noise == 'on':
        beta, gamma = [random.gauss(0, math.sqrt(timestep)) for i in range(2)]
        omega = numpy.random.normal(0, math.sqrt(timestep),(G.shape[1],1))
        ksi = numpy.random.normal(0, math.sqrt(timestep),(D.shape[1],1))
    elif noise == 'off':
        beta, gamma = [random.gauss(0, 0) for i in range(2)]
        omega = numpy.random.normal(0, 0,(G.shape[1],1))
        ksi = numpy.random.normal(0, 0,(D.shape[1],1))
    else:
        raise NotImplementedError
    X = X + (A@X + B*U[i,nt])*timestep + F@X*beta + Y*U[i,nt]*gamma + G@omega
    dy =  C@X*timestep + D@ksi
    Xhat = Xhat + (Ac@Xhat + Bc*U[i,nt])*timestep + K@(dy - Cc@Xhat*timestep)
    Mov[i+1,nt,:,0] = X.reshape(1,-1)
    Mov[i+1,nt,:,1] = Xhat.reshape(1,-1)
    return Mov, U
