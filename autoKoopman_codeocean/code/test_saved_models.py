import autokoopman
from autokoopman import auto_koopman
from symbolic import bio2, fhn, lalo20, prde20, robe21, spring, pendulum, trn_constants, pendulum_withD_oncontrol, pendulum_withD_onfullstate, \
    inverted_pendulum_withD_oncontrol, inverted_pendulum_withD_onfullstate

# from autokoopman.benchmark.bio2 import Bio2 as bio2
# from autokoopman.benchmark.fhn import FitzHughNagumo as fhn
# from autokoopman.benchmark.lalo20 import LaubLoomis as lalo20
# from autokoopman.benchmark.prde20 import ProdDestr as prde20
# from autokoopman.benchmark.robe21 import RobBench as robe21
# from autokoopman.benchmark.spring import Spring as spring
# from autokoopman.benchmark.pendulum import PendulumWithInput as pendulum

# import autokoopman.benchmark.bio2 as bio2
# import autokoopman.benchmark.fhn  as fhn
# import autokoopman.benchmark.lalo20  as lalo20
# import autokoopman.benchmark.prde20  as prde20
# import autokoopman.benchmark.robe21  as robe21
# import autokoopman.benchmark.spring  as spring
# import autokoopman.benchmark.pendulum  as pendulum

import numpy as np 
import pickle 
import pdb 

import sys
sys.path.append(".")

filename = "/Users/nikhilushinde/Documents/Grad/research/arclab/AutoKoopman/test_saved_models/experimentsSymbolic_modified_model_results_withD.pickle"

class loadedModels(): 
    """
    Pickle file containing following dictionary structure: 
        - key: benchmark name: 
        - value: dict
            - key: observable type ('id', 'poly', 'rff')
            - value: dict: 
                - key: A
                    - Value: A matrix: Ax + Bu
                - key: B: 
                    - Value: B matrix: Ax + Bu
                - key: hyperparameters: 
                    - value: list of strings indicating hyperparameters
                - key: hyperparameter_values: 
                    - value: list of values corresponding to the hyperparameters
                - key: estimator: 
                    - value: optimized estimator object from autokoopman
                - key: samp_period:
                    - value: sampling period to generate the data 
    """
    def __init__(self, filename): 
        """
        args: 
            - filename: filename containing above structure to load
        """
        with open(filename, "rb") as handle: 
            self.saved_models_dict = pickle.load(handle)

        inverted_pendulum_samp_period = 0.1

        # NOTE: make sure these base models are setup the same way during data generation
        self.benches = [pendulum.PendulumWithInput(beta=0.05), spring.Spring(), fhn.FitzHughNagumo(), robe21.RobBench(), prde20.ProdDestr(), lalo20.LaubLoomis(), bio2.Bio2(), trn_constants.TRNConstants()]
        self.benches.extend([pendulum_withD_oncontrol.PendulumWithInputAndDisturbcontrol(beta=0.05), pendulum_withD_onfullstate.PendulumWithInputAndDisturbcontrol(beta=0.05)])
        self.benches.extend([inverted_pendulum_withD_oncontrol.InvertedPendulumWithInputAndDisturbcontrol(beta=-0.05, samp_period=inverted_pendulum_samp_period), 
                             inverted_pendulum_withD_onfullstate.InvertedPendulumWithInputAndDisturbcontrol(beta=-0.05, samp_period=inverted_pendulum_samp_period)])

        self.benches_names = [bench.name for bench in self.benches]

        self.benchmark_name = None
        self.observable = None
        self.curr_dict = None
        self.curr_benchmark_obj = None
        return 

    def check_availability(self): 
        """
        prints available configurations
        """
        print("Configurations available: ")
        print("benchmarks: ", list(self.saved_models_dict.keys()))
        print("observable examples: ", list(self.saved_models_dict[list(self.saved_models_dict.keys())[0]].keys()))
        print()
        return 

    def check_benchmark(self, benchmark_name):
        """
        Checks the available observables for a benchmark
        """
        if benchmark_name not in self.benches_names: 
            print(benchmark_name + " : NOT AVAILABLE: try benchmark")
            self.check_availability()
            print()
            return 

        print("Observables available for benchmark: ", benchmark_name)
        print(list(self.saved_models_dict[benchmark_name].keys()))
        print()
        return 


    def check_state_space(self):
        """
        print the current state space for the set benchmark
        """
        print("Variables for benchmark model: " + str(self.benchmark_name) + " : ", self.curr_benchmark_obj.names)
        print()
        return 

    def is_setup(self): 
        """
        check if configured
        return: 
            - boolean indicator: True if setup is done else false
        """
        if self.benchmark_name is None or self.observable is None or self.curr_dict is None: 
            print("WARNING: Set benchmark and observable before using any function")
            print()
            return False 
        return True

    def set(self, benchmark_name, observable): 
        """
        Sets which benchmark and observable to use from the loaded configurations
        """
        
        try: 
            self.benchmark_name = benchmark_name
            self.observable = observable 

            self.curr_dict = self.saved_models_dict[self.benchmark_name][self.observable]
        except: 
            print("Incorrect benchmark or observable names provided try:")
            print("benchmarks: ", list(self.saved_models_dict.keys()))
            print("observable examples: ", list(self.saved_models_dict[list(self.saved_models_dict.keys())[0]].keys()))
            raise NotImplementedError

        bench_idx = self.benches_names.index(self.benchmark_name)
        self.curr_benchmark_obj = self.benches[bench_idx]

    def get_matrices(self):
        """
        returns: 
            - 
        """ 
        if not self.is_setup(): 
            return None, None
        
        A = self.curr_dict["A"]
        B = self.curr_dict["B"]

        return A, B 

    def lift_data(self, data): 
        """
        Lifts the data using the optimized observables
        args: 
            - data: numpy array: (num data, data dim)
        returns: 
            - lifted_data: numpy array: (lifted dim, num data)
        """
        if not self.is_setup(): 
            return None

        estimator = self.curr_dict["estimator"]
        return estimator.obs.obs_fcn(data)

    def generate_gt_traj(self, initial_condition, time_span=None, inputs=None, time_eval=None): 
        """
        Generate ground truth data with the benchmark that you have set
        args: 
            - initial_condition: the initial state 
            - time_span: optional: 
                - None: controlled system 
                - list: for autonomous system [start time, end time]
            - inputs: optional
                - None: autonomous system 
                - numpy array: the inputs to apply to the system at each timestep
            - time_eval: optional:
                - None: autonomous system 
                - numpy array: numpy array containing all the time points that you want to evaluate at
                - NOTE: must be the same length as the inputs
        returns; 
            - data: numpy array of the trajectory 
        """
        if not self.is_setup(): 
            return None
        
        if (time_span is None and inputs is None and time_eval is None) or ((time_span is not None and inputs is not None and time_eval is not None)): 
            print("WARNING: Need to specify timespan for autonomous systems OR inputs and time_eval for controlled systems")
            return None 

        if inputs is None:     
            # autonomous system     
            traj = self.curr_benchmark_obj.solve_ivp(
                initial_state=initial_condition, 
                tspan=time_span, 
                sampling_period=self.curr_dict["samp_period"]
            )
        else: 
            if len(inputs) != len(time_eval): 
                print("WARNING: need same number of inputs as time_eval points")
                return None

            # controlled system
            traj = self.curr_benchmark_obj.solve_ivp(
                initial_state=initial_condition, 
                sampling_period=self.curr_dict["samp_period"],
                inputs=inputs,
                teval=time_eval, 
            )
        return traj

if __name__ == "__main__": 
    testobj = loadedModels(filename=filename)

    pdb.set_trace()
    print()