import numpy as np
import random
import scipy
import matplotlib.pyplot as plt
import os.path
import pandas as pd

from autokoopman import auto_koopman
from autokoopman.core.observables import IdentityObservable, RFFObservable
from autokoopman.estimator.koopman import KoopmanContinuousEstimator, KoopmanDiscEstimator
import autokoopman.core.trajectory as traj

PATH = '/data/'

def load_data(benchmark):
    """load the measured data"""

    path = os.path.join(PATH, benchmark)
    cnt = 1
    data = []

    while True:
        dirpath = os.path.join(path, 'measurement_' + str(cnt))
        if os.path.isdir(dirpath):
            states = np.asarray(pd.read_csv(os.path.join(dirpath, 'trajectory.csv')))
            inputs = np.asarray(pd.read_csv(os.path.join(dirpath, 'input.csv')))
            time = np.asarray(pd.read_csv(os.path.join(dirpath, 'time.csv')))
            time = np.resize(time, (time.shape[0],))
            data.append(traj.Trajectory(time[:-1], states[:-1, :], inputs))
            cnt += 1
        else:
            break

    if len(data) > 100:
        data = data[0:100]

    #data = data[0:1]

    ids = np.arange(0, len(data)).tolist()
    data = traj.TrajectoriesData(dict(zip(ids, data)))

    return data

def train_model_cont(data):
    """identify a continuous-time model"""

    dt = data._trajs[0].times[1] - data._trajs[0].times[0]
    data = data.interp_uniform_time(dt)
    cont = KoopmanContinuousEstimator(
        IdentityObservable() | RFFObservable(3, 200, 0.0001), 3, 3
    )
    cont.fit(data)

    return cont.model, cont

def train_model_disc(data):
    """identify a discrete-time model"""

    dt = data._trajs[0].times[1] - data._trajs[0].times[0]
    data = data.interp_uniform_time(dt)
    cont = KoopmanDiscEstimator(
        IdentityObservable() | RFFObservable(3, 200, 0.0001), dt, 3, 3
    )
    cont.fit(data)

    return cont.model, cont

def simulate_disc(model, x0, u):
    """simulate a trajectory for a discrete-time model"""

    x0_ = model.obs.obs_fcn(x0)
    X = np.zeros((x0_.shape[0], u.shape[1] + 1))
    X[:, [0]] = x0_

    for i in range(0, u.shape[1]):
        X[:, [i+1]] = model._A @ X[:, [i]] + model._B @ u[:, [i]]

    return X[0:len(x0), :]

def simulate_cont(model, x0, u, dt):
    """simulate a trajectory for a continuous-time model"""

    A, B = conversion_discrete_time(model._A, model._B, dt)

    x0_ = model.obs.obs_fcn(x0)
    X = np.zeros((x0_.shape[0], u.shape[1] + 1))
    X[:, [0]] = x0_

    for i in range(0, u.shape[1]):
        X[:, [i + 1]] = A @ X[:, [i]] + B @ u[:, [i]]

    return X[0:len(x0), :]


def conversion_discrete_time(A, B, dt):
    """convert a continuous time linear system to an equivalent linear system"""

    # system matrix
    A_ = scipy.linalg.expm(A * dt)

    # input matrix
    temp1 = np.eye(A.shape[0]) * dt
    temp2 = temp1
    cnt = 2

    while 1:
        temp1 = np.dot(temp1 * dt / cnt, A)
        temp2 = temp2 + temp1
        cnt = cnt + 1
        if (abs(temp1) < np.finfo(float).eps).all() | cnt > 1000:
            break

    B_ = np.dot(temp2, B)

    return A_, B_


if __name__ == '__main__':

    # initialization
    benchmark = 'ElectricCircuit'

    # load data
    data = load_data(benchmark)

    # train the Koopman model
    model_cont, estimator_cont = train_model_cont(data)
    model_disc, estimator_disc = train_model_disc(data)

    # loop over all test trajectories
    traj = data._trajs[4]

    # simulate using the learned model
    iv = traj.states[0, :]
    dt = traj.times[1] - traj.times[0]

    traj_cont = simulate_cont(estimator_cont, iv, traj.inputs.T, dt)
    traj_disc = simulate_disc(estimator_disc, iv, traj.inputs.T)

    # visualization
    plt.plot(traj.times, traj.states[:, 2], 'r', label='original data')
    plt.plot(traj.times, traj_disc[2, 1:], 'b', label='discrete time')
    plt.plot(traj.times, traj_cont[2, 1:], 'g', label='continuous time')
    plt.legend()
    plt.xlim((0, 0.8))
    plt.xlabel("time")
    plt.ylabel("output voltage")
    plt.savefig("../results/figure4.pdf")
