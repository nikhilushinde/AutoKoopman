from scipy.io import savemat
import numpy as np
import os.path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from autokoopman import auto_koopman
import autokoopman.core.trajectory as traj

def load_training_data():
    """load the measured training data"""

    # load data measured from a F1tenth racecar
    path = '/data/F1tenthCar'
    cnt = 1
    data = []
    ids = []

    while True:
        dirpath = os.path.join(path, 'measurement_' + str(cnt))
        if os.path.isdir(dirpath):
            states = np.asarray(pd.read_csv(os.path.join(dirpath, 'trajectory.csv')))
            inputs = np.asarray(pd.read_csv(os.path.join(dirpath, 'input.csv')))
            time = np.asarray(pd.read_csv(os.path.join(dirpath, 'time.csv')))
            time = np.resize(time, (time.shape[0],))
            names = ['x', 'y', 'phi', 'v']
            names_inputs = ['steer', 'speed']
            data.append(traj.Trajectory(time[:-1], states[:-1, :], inputs, names, names_inputs))
            ids.append(str(cnt))
            cnt += 1
        else:
            break

    return traj.TrajectoriesData(dict(zip(ids, data)))


def train_model():
    """train the Koopman model using the AutoKoopman library"""

    # get training data
    training_data = load_training_data()

    # learn model from data
    experiment_results = auto_koopman(
        training_data,  # list of trajectories
        n_splits=5,
        opt='grid',
        n_obs=20,
        lengthscale=(0.1, 2.0),
        rank=(1, 10, 1),
        grid_param_slices=3
    )

    # debug
    # get the model from the experiment results
    model = experiment_results['tuned_model']

    # simulate using the learned model
    tmp = list(training_data._trajs.values())

    for t in tmp:
        iv = t.states[0, :]
        start_time = t.times[0]
        end_time = t.times[len(t.times)-1]
        teval = np.linspace(start_time, end_time, len(t.times))

        trajectory = model.solve_ivp(
            initial_state=iv,
            tspan=(start_time, end_time),
            sampling_period=t.times[1] - t.times[0],
            inputs=t.inputs,
            teval=teval
        )

        plt.plot(t.states[:, 0], t.states[:, 1], 'r')
        plt.plot(trajectory.states[:, 0], trajectory.states[:, 1], 'b')

    plt.show()

    # get the model from the experiment results
    return experiment_results['estimator']


if __name__ == '__main__':

    # train the Koopman model
    model = train_model()

    # save data from the Koopman model
    D = model.obs.observables[1].D
    u = model.obs.observables[1].u
    w = model.obs.observables[1].w
    A = model._A
    B = model._B

    data = {'A': A, 'B': B, 'D': D, 'u': u, 'w': w}
    savemat('/code/control/KoopmanModel.mat', data)

