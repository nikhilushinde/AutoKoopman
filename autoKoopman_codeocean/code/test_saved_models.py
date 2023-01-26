import autokoopman
from symbolic import bio2, fhn, lalo20, prde20, robe21, spring, pendulum, trn_constants

import numpy as np 
import pickle 
import pdb 

import sys
sys.path.append(".")

filename = "/Users/nikhilushinde/Documents/Grad/research/arclab/AutoKoopman/autoKoopman_codeocean/results/TESTS.pickle"



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


        # NOTE: make sure these base models are setup the same way during data generation
        self.benches = [pendulum.PendulumWithInput(beta=0.05), spring.Spring(), fhn.FitzHughNagumo(), robe21.RobBench(), prde20.ProdDestr(), lalo20.LaubLoomis(), bio2.Bio2(), trn_constants.TRNConstants()]
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

    def check_current_state_space(self):
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

    def set_benchmark_observable(self, benchmark_name, observable): 
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

    def generate_gt_traj(self, initial_condition, time_span): 
        """
        Generate ground truth data with the benchmark that you have set
        args: 
            - initial_condition: the initial state 
            - time_span: [start time, end time]
        returns; 
            - data: numpy array of the trajectory 
        """
        if not self.is_setup(): 
            return None

        traj = self.curr_benchmark_obj.solve_ivp(
            initial_state=initial_condition, 
            tspan=time_span, 
            sampling_period=self.curr_dict["samp_period"]
        )
        return traj

if __name__ == "__main__": 
    testobj = loadedModels(filename=filename)

    pdb.set_trace()
    print()