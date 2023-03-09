import matplotlib.pyplot as plt
import numpy as np
# this is the convenience function
import torch

import sys 
sys.path.append("../../..")
from autokoopman import auto_koopman

# for a complete example, let's create an example dataset using an included benchmark system
from symbolic import bio2, fhn, lalo20, prde20, robe21, spring, pendulum, trn_constants, pendulum_withD_oncontrol, pendulum_withD_onfullstate, \
    inverted_pendulum_withD_oncontrol, inverted_pendulum_withD_onfullstate, vdp_py, duff_py
from auxiliary.glop import Glop
import random
import copy

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from numpy.linalg import norm
import statistics
import os
import csv
import time
import sys

import pickle 
import pdb


def get_training_data(bench, param_dict):
    init_states = get_init_states(bench, param_dict["train_size"])
    if bench._input_vars:
        steps = []
        if type(bench.input_type) != type([]): 
            input_type = [bench.input_type]
        else: 
            input_type = bench.input_type

        for num, (low, high) in enumerate(zip(bench.input_set_low, bench.input_set_high)):
            if input_type[num] == "step":
                params = np.random.uniform(low, high, size=(param_dict["train_size"], 3))
                curr_steps = [make_input_step(*p, bench.teval) for p in params]
            elif input_type[num] == "rand":
                curr_steps = [make_random_input(low, high, bench.teval) for i in range(param_dict["train_size"])]
            else:
                sys.exit("Please set an input type for your benchmark")
            steps.append(np.array(curr_steps))
        
        if len(steps) > 1: 
            steps = np.stack(tuple(steps), axis=-1)
        else: 
            steps = steps[0]
        training_data = bench.solve_ivps(initial_states=init_states, inputs=steps, teval=bench.teval)
    else:
        training_data = bench.solve_ivps(initial_states=init_states, tspan=[0.0, 10.0],
                                         sampling_period=param_dict["samp_period"])

    return training_data


def get_init_states(bench, size, init_seed=0):
    if hasattr(bench, 'init_constrs'):
        init_states = []
        for i in range(size):
            init_state_dict = glop_init_states(bench, i + init_seed)
            init_state = []
            for name in bench.names:
                init_state.append(init_state_dict[name])
            init_states.append(init_state)
        init_states = np.array(init_states)
    else:
        init_states = np.random.uniform(low=bench.init_set_low,
                                        high=bench.init_set_high, size=(size, len(bench.names)))

    return init_states


def glop_init_states(bench, seed):
    constrs = []
    for constr in bench.init_constrs:
        constrs.append(constr)
    for i, (name, init_low, init_high) in enumerate(zip(bench.names, bench.init_set_low, bench.init_set_high)):
        low_constr = f"{name} >= {init_low}"
        high_constr = f"{name} <= {init_high}"
        constrs.extend([low_constr, high_constr])

    glop = Glop(bench.names, constrs)
    pop_item = random.randrange(len(bench.names))
    names, init_set_low, init_set_high = copy.deepcopy(bench.names), copy.deepcopy(bench.init_set_low), copy.deepcopy(
        bench.init_set_high)
    names.pop(pop_item)
    init_set_low.pop(pop_item)
    init_set_high.pop(pop_item)
    for i, (name, init_low, init_high) in enumerate(zip(names, init_set_low, init_set_high)):
        glop.add_tend_value_obj_fn(name, [init_low, init_high], seed + i)

    glop.minimize()

    sol_dict = glop.get_all_sols()
    return sol_dict


def get_trajectories(bench, iv, samp_period):
    # get the model from the experiment results
    model = experiment_results['tuned_model']

    if bench._input_vars:
        if type(bench.input_type) != type([]): 
            input_type = [bench.input_type]
        else: 
            input_type = bench.input_type

        test_traj_length = len(bench.teval) 
        test_inp = []
        for num in range(len(input_type)): 
            if input_type[num] == "step":
                curr_test_inp = np.sin(np.linspace(0, 10, test_traj_length)) # NOTE: following their placeholder for actual inputs
            if input_type[num] == "rand": 
                low = bench.input_set_low[num]
                high = bench.input_set_high[num]
                curr_test_inp = make_random_input(low, high, bench.teval) 

            test_inp.append(curr_test_inp)
            
        if len(test_inp) > 1: 
            test_inp = np.stack(tuple(test_inp), axis=-1)
        else: 
            test_inp = test_inp[0]

        # simulate using the learned model
        trajectory = model.solve_ivp(
            initial_state=iv,
            inputs=test_inp,
            teval=bench.teval,
        )
        # simulate the ground truth for comparison
        true_trajectory = bench.solve_ivp(
            initial_state=iv,
            inputs=test_inp,
            teval=bench.teval,
        )

    else:
        # simulate using the learned model
        trajectory = model.solve_ivp(
            initial_state=iv,
            tspan=(0.0, 10.0),
            sampling_period=samp_period
        )
        # simulate the ground truth for comparison
        true_trajectory = bench.solve_ivp(
            initial_state=iv,
            tspan=(0.0, 10.0),
            sampling_period=samp_period
        )

    return trajectory, true_trajectory


def test_trajectories(bench, num_tests, samp_period):
    # euc_norms = []
    # for j in range(num_tests):
    #     iv = get_init_states(bench, 1, j + 10000)[0]
    #     try:
    #         trajectory, true_trajectory = get_trajectories(bench, iv, samp_period)
    #         y_true = np.matrix.flatten(true_trajectory.states)
    #         y_pred = np.matrix.flatten(trajectory.states)
    #         euc_norm = norm(y_true - y_pred) / norm(y_true)
    #         euc_norms.append(euc_norm)

    #     except:
    #         print("ERROR--solve_ivp failed (likely unstable model)")
    #         euc_norms.append(np.infty)

    # return statistics.mean(euc_norms)

    euc_norms = []
    for j in range(num_tests):
        iv = get_init_states(bench, 1, j + 10000)[0]
        trajectory, true_trajectory = get_trajectories(bench, iv, samp_period)

        # plot function
        # if j  <= 10: 
        #     ytrue_plot = true_trajectory.states
        #     ypred_plot = trajectory.states
        #     plt.plot(ytrue_plot[:, 0], ytrue_plot[:, 1], "red")
        #     plt.plot(ypred_plot[:, 0], ypred_plot[:, 1], "blue")
        #     plt.show(block=True)

        y_true = np.matrix.flatten(true_trajectory.states)
        y_pred = np.matrix.flatten(trajectory.states)
        euc_norm = norm(y_true - y_pred) / norm(y_true)
        euc_norms.append(euc_norm)

    return statistics.mean(euc_norms)



def make_input_step(duty, on_amplitude, off_amplitude, teval):
    """produce a step response input signal for the pendulum"""
    length = len(teval)
    inp = np.zeros((length,))
    phase_idx = int(length * duty)
    inp[:phase_idx] = on_amplitude
    inp[phase_idx:] = off_amplitude
    return inp


def make_random_input(low, high, teval):
    length = len(teval)
    inp = np.zeros((length,))
    for i in range(len(inp)):
        inp[i] = np.random.uniform(low, high)
    return inp


def store_data(row, filename='symbolic_data.csv'):
    with open(f'../results/{filename}', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def store_data_heads(row, filename='symbolic_data.csv'):
    with open(f'../results/{filename}', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def plot(trajectory, true_trajectory, var_1, var_2):
    plt.figure(figsize=(10, 6))
    # plot the results
    if var_2 == -1:  # plot against time
        plt.plot(trajectory.states[:, var_1], label='Trajectory Prediction')
        plt.plot(true_trajectory.states[:, var_1], label='Ground truth')
    else:
        plt.plot(trajectory.states.T[var_1], trajectory.states.T[var_2], label='Trajectory Prediction')
        plt.plot(true_trajectory.states.T[var_1], true_trajectory.states.T[var_2], label='Ground Truth')

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.grid()
    plt.legend()
    plt.title("Bio2 Test Trajectory Plot")
    plt.show()


def plot_trajectory(bench, var_1=0, var_2=-1, seed=100):
    iv = get_init_states(bench, 1, seed)[0]
    trajectory, true_trajectory = get_trajectories(bench, iv, param_dict["samp_period"])
    plot(trajectory, true_trajectory, var_1, var_2)

def test_plot(model): 
    test_inp = np.ones(200)
    test_inp[:10] = 0.1
    test_inp[10:] = 0 

    teval = np.arange(200) * 0.1

    trajectory = model.solve_ivp([0, 0], inputs=test_inp, teval=teval)
    states = trajectory.states

    plt.plot(states[:, 0], states[:, 1])
    plt.show(block=True)


if __name__ == '__main__':
    all_matrices = {}

    save = True

    inverted_pendulum_samp_period = 0.1

    benches = []
    # our models 
    # benches.extend([inverted_pendulum_withD_oncontrol.InvertedPendulumWithInputAndDisturbcontrol(beta=-0.05, samp_period=inverted_pendulum_samp_period), 
                    # inverted_pendulum_withD_onfullstate.InvertedPendulumWithInputAndDisturbcontrol(beta=-0.05, samp_period=inverted_pendulum_samp_period)])
    # benches.extend([pendulum_withD_oncontrol.PendulumWithInputAndDisturbcontrol(beta=0.05), pendulum_withD_onfullstate.PendulumWithInputAndDisturbcontrol(beta=0.05)])
    # benches.extend([pendulum.PendulumWithInput(beta=0.05), spring.Spring(), fhn.FitzHughNagumo(), robe21.RobBench(), prde20.ProdDestr(), lalo20.LaubLoomis(), bio2.Bio2(), trn_constants.TRNConstants()])
    benches.extend([duff_py.Duffing(), vdp_py.VanderPol()])

    # their models
    # benches = [pendulum.PendulumWithInput(beta=0.05), pendulum_withD_oncontrol.PendulumWithInputAndDisturbcontrol(beta=0.05), pendulum_withD_onfullstate.PendulumWithInputAndDisturbcontrol(beta=0.05)]#[pendulum.PendulumWithInput(beta=0.05), spring.Spring(), fhn.FitzHughNagumo(), robe21.RobBench(), prde20.ProdDestr(), lalo20.LaubLoomis(), bio2.Bio2(), trn_constants.TRNConstants()]
    # benches = [pendulum_withD_oncontrol.PendulumWithInputAndDisturbcontrol(beta=0.05), pendulum_withD_onfullstate.PendulumWithInputAndDisturbcontrol(beta=0.05)]#[pendulum.PendulumWithInput(beta=0.05), spring.Spring(), fhn.FitzHughNagumo(), robe21.RobBench(), prde20.ProdDestr(), lalo20.LaubLoomis(), bio2.Bio2(), trn_constants.TRNConstants()]
    # benches = [pendulum_withD_onfullstate.PendulumWithInputAndDisturbcontrol(beta=0.05)]
    obs_types = ['id', 'poly', 'rff']#['id', 'poly', 'rff', 'deep']
    if save: 
        store_data_heads(["", ""] + ["euc_norm", "time(s)", ""] * len(obs_types))
        # save_filename = "../results/experimentsSymbolic_modified_model_results_withD.pickle"
        # save_filename = "../results/TESTS.pickle"
        # save_filename = "../results/experimentsSymbolic_inverted_pend_10timesteps_0p1samp_0p5init.pickle"
        save_filename = "../results/experimentsSymbolic_vdp_duff.pickle"

        print("Going to save: ", save_filename)
        
    print("\n\n\nStarting experiments: \n\n\n")
    for benchmark in benches:
        result = [benchmark.name, ""]

        all_matrices[benchmark.name] = {}

        for obs in obs_types:

            all_matrices[benchmark.name][obs] = {}

            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            if obs == 'deep':
                opt = 'bopt'
            else:
                opt = 'grid'
            param_dict = {"train_size": 10, "samp_period": 0.1, "obs_type": obs, "opt": opt, "n_obs": 200,
                          "grid_param_slices": 5, "n_splits": 5, "rank": (1, 200, 40), "verbose": False}

            # have the inverted pendulum have a smaller samp_period 
            if "inverted_pendulum" in benchmark.name: 
                param_dict["samp_period"] = inverted_pendulum_samp_period

            # generate training data
            training_data = get_training_data(benchmark, param_dict)
            start = time.time()
            # learn model from data
            experiment_results = auto_koopman(
                training_data,  # list of trajectories
                sampling_period=param_dict["samp_period"],  # sampling period of trajectory snapshots
                obs_type=param_dict["obs_type"],  # use Random Fourier Features Observables
                opt=param_dict["opt"],  # grid search to find best hyperparameters
                n_obs=param_dict["n_obs"],  # maximum number of observables to try
                max_opt_iter=200,  # maximum number of optimization iterations
                grid_param_slices=param_dict["grid_param_slices"],
                # for grid search, number of slices for each parameter
                n_splits=param_dict["n_splits"],  # k-folds validation for tuning, helps stabilize the scoring
                rank=param_dict["rank"]  # rank range (start, stop, step) DMD hyperparameter
            )
            end = time.time()

            # save the matrices 
            all_matrices[benchmark.name][obs]['A'] = experiment_results['estimator']._A
            all_matrices[benchmark.name][obs]['B'] = experiment_results['estimator']._B
            all_matrices[benchmark.name][obs]['hyperparameters'] = experiment_results['hyperparameters']
            all_matrices[benchmark.name][obs]['hyperparameter_values'] = experiment_results['hyperparameter_values']
            all_matrices[benchmark.name][obs]['estimator'] = experiment_results['estimator']
            all_matrices[benchmark.name][obs]['samp_period'] = param_dict['samp_period']
            # all_matrices[benchmark.name][obs]['tuned_model'] = experiment_results['tuned_model']

            print()
            print("A shape: ", experiment_results['estimator']._A.shape)
            print("B shape: ", experiment_results['estimator']._B.shape)
            print()

            euc_norm = test_trajectories(benchmark, 10, param_dict["samp_period"])
            comp_time = round(end - start, 3)

            print(benchmark.name)
            print(f"observables type: {obs}")
            print(f"The average euc norm perc error is {round(euc_norm * 100, 2)}%")
            print("time taken: ", comp_time)
            # store and print results
            result.append(round(euc_norm * 100, 2))
            result.append(comp_time)
            result.append("")

            if save:
                # save results until now
                print("Saving! ", save_filename)
                with open(save_filename, 'wb') as handle:
                    pickle.dump(all_matrices, handle)

        if save: 
            store_data(result)

