# the notebook imports
import csv
import os
import time
import random

import torch
import matplotlib.pyplot as plt
import numpy as np

# this is the convenience function
from autokoopman import auto_koopman

import autokoopman.benchmark.fhn as fhn
import autokoopman.core.trajectory as traj
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.linalg import norm
import statistics


def get_train_data(filepath):
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, filepath)
    training_data = traj.TrajectoriesData.from_csv(filename)

    return training_data, dirname


def get_true_trajectories(filepath):
    # get the model from the experiment results
    model = experiment_results['tuned_model']
    # simulate using the learned model
    filename = os.path.join(dirname, filepath)
    true_trajectories = traj.TrajectoriesData.from_csv(filename)

    return true_trajectories, model


def test_trajectories(true_trajectories, model, tspan):
    euc_norms = []
    for i in range(9):
        init_s = true_trajectories[i].states[0]
        iv = init_s
        try:
            trajectory = model.solve_ivp(
                initial_state=iv,
                tspan=tspan,
                sampling_period=0.1
            )
            y_true = np.matrix.flatten(true_trajectories[i].states)
            y_pred = np.matrix.flatten(trajectory.states)
            euc_norm = norm(y_true - y_pred) / norm(y_true)
            euc_norms.append(euc_norm)

        except:
            print("ERROR--solve_ivp failed (likely unstable model)")
            euc_norms.append(np.infty)

    return statistics.mean(euc_norms)


def store_data(row, filename='black_box_data.csv'):
    with open(f'../results/{filename}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def store_data_heads(row, filename='black_box_data.csv'):
    with open(f'../results/{filename}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)


if __name__ == '__main__':

    benches = ["Engine Control", "Longitudunal", "Ground_collision"]
    train_datas = ['../data/F16Aircraft/trainingdata/engine/trainingDataUniform20.csv',
                   '../data/F16Aircraft/trainingdata/long/trainingDataUniform20.csv',
                   '../data/F16Aircraft/trainingdata/gsac/trainingDataUniform400.csv']
    tspans = [(0.0, 60), (0.0, 15), (0.0, 15)]
    trajectories_filepaths = ['../data/F16Aircraft/testdata/checkEngine.csv', '../data/F16Aircraft/testdata/long.csv', '../data/F16Aircraft/testdata/gsac.csv']
    obs_types = ['id', 'poly', 'rff', 'deep']
    store_data_heads(["", ""] + ["euc_norm", "time(s)", ""] * len(obs_types))
    for benchmark, train_data, tspan, trajectories_filepath in zip(benches, train_datas, tspans,
                                                                   trajectories_filepaths):
        result = [benchmark, ""]
        for obs in obs_types:
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            training_data, dirname = get_train_data(train_data)
            start = time.time()

            if obs == 'deep':
                opt = 'bopt'
            else:
                opt = 'grid'
            # learn model from data
            experiment_results = auto_koopman(
                training_data,  # list of trajectories
                sampling_period=0.1,  # sampling period of trajectory snapshots
                obs_type=obs,  # use Random Fourier Features Observables
                opt=opt,  # grid search to find best hyperparameters
                n_obs=200,  # maximum number of observables to try
                max_opt_iter=200,  # maximum number of optimization iterations
                grid_param_slices=5,  # for grid search, number of slices for each parameter
                n_splits=5,  # k-folds validation for tuning, helps stabilize the scoring
                rank=(1, 200, 40),  # rank range (start, stop, step) DMD hyperparameter
                verbose=False
            )
            end = time.time()
            true_trajectories, model = get_true_trajectories(trajectories_filepath)
            euc_norm = test_trajectories(true_trajectories, model, tspan)
            comp_time = round(end - start, 3)

            print(benchmark)
            print(f"observables type: {obs}")
            print(f"The average euc norm perc error is {round(euc_norm * 100, 2)}%")
            print("time taken: ", comp_time)
            # store and print results
            result.append(round(euc_norm * 100, 2))
            result.append(comp_time)
            result.append("")

        store_data(result)