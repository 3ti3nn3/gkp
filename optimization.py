# import packages
import numpy as np
import strawberryfields as sf
import scipy
from joblib import Parallel, delayed
import pickle
import os


# set hyperparametsr 
sf.hbar = 1 # Setting convention for hbar
backend = "fock"


# content
class Optimization: 

    cutoff     = 10 # Truncation of the Fock space
    delta      = 0.5 # Inverse width of Gaussian envelope
    epsilon    = 0.5**2 # Strawberry Fields parameter epsilon which defines the Gaussian envelope in sf.ops.GKP
    num_blocks = 6 # Number of repeated blocks of gates in circuit
    num_trials = 2 # Number of repeated optimization trials
    n_jobs     = 2 # For parallelisation - best if this is a factor of num_trials
    state      = [0,0]

    # Optimisation function 
    circuit = ''

    # Optimization results
    Fids    = None
    Sols    = None
    bestFid = None
    bestSol = None

    # Gates
    gates = None 
    
    # initializer
    def __init__(self, gates=None, circuit=None, cutoff=None, delta=None, epsilon=None, num_blocks=None, num_trials=None, n_jobs=None, state=None, file_name=None): 
        
        # if a new class is initialized
        if file_name is None: 
            self.cutoff     = cutoff
            self.delta      = delta
            self.epsilon    = epsilon
            self.num_blocks = num_blocks
            self.num_trials = num_trials
            self.n_jobs     = n_jobs
            self.state      = state

            self.Fids    = None
            self.Sols    = None
            self.bestFid = None
            self.bestSol = None

            self.gates   = gates
            self.circuit = circuit

        # if the date is loaded from a file
        elif type(file_name) == str: 
            with open('data/' + file_name, 'rb') as file:
                saved_dict = pickle.load(file)
            self.__dict__.update(saved_dict)

        # cause error
        else: 
            raise ValueError(f'"{file_name}" is an invalid file name!')


    # access best fidelity
    def get_bestFid(self): 
        return self.bestFid


    # access best solutions for functions parameters
    def get_bestSol(self):
        return self.bestSol


    # access fidelity
    def get_Fids(self): 
        return self.Fids


    # access solutions for functions parameters
    def get_Sols(self):
        return self.Sols


    # access circuit name
    def get_circuit(self):
        return self.circuit


    # access gates 
    def get_gates(self):
        return self.gates


    # access cutoff 
    def get_cutoff(self):
        return self.cutoff 


    # access delta
    def get_delta(self):
        return self.delta


    # acess epsilon 
    def get_epsilon(self):
        return self.epsilon


    # access num_trials
    def get_num_trials(self):
        return self.num_trials


    # access state
    def get_state(self):
        return self.state


    # access num_blocks
    def get_num_blocks(self):
        return self.num_blocks


    # save the whole class
    def save(self, file_name):
        os.makedirs('data', exist_ok=True)
        with open('data/'+file_name, 'wb') as file: 
            pickle.dump(self.__dict__, file)
        return 


    # define frontend function for cooled_squared_vac_overlap_complex
    def cooled_squared_vac_overlap(self, params):
        '''
        Return the squared overlap with vacuum of the state resulting from application of 
        the parametrized circuit to the (given) approximate GKP state of given epsilon.
    
        The parametrized circuit consists of len(params)//4 blocks composed of:
            - 2 orthogonal displacements
            - one Kerr gate
            - one squeezing gate (with real squeezing parameter).
    
        The number of blocks is guessed from the size of the input parameters, assumed to be a multiple of 4.
    
        Parameters: 
            - params (array): Gate parameters of the applied circuit. 
    
        Returns: 
            - float: negative fidelity
        '''
        # Initialize Strawberry Fields program
        prog = sf.Program(1) # Photonic circuit with one mode
        
        with prog.context as q: 
            # Initialize state to be cooled down (target)
            sf.ops.GKP(epsilon=self.epsilon, state=self.state) | q 

            for i in range(len(params)//len(self.gates)):
                for k in range(len(self.gates)): 
                    self.gates[k](params[i*len(self.gates)+k]) | q

        # Initialize engine, selecting Fock backend (cutoff is hyperparameter of the file)
        eng = sf.Engine(backend, backend_options={"cutoff_dim": self.cutoff}) 
    
        # Execute program on engine
        cooled_state = eng.run(prog, shots=1).state # output state of parametrized circuit
        
        # Return fidelity with vacuum (negative for minimization later)
        return -cooled_state.fock_prob([0])


    # parallel trials for optimizing the gate parameters to maximize fidelity
    def parallel_optimize_circuit(self, args=()):
        """
        Parallel optimization of gate parameters and maximzation of fidelity.
    
        Parameters:
            - x0 (variable): Initial guess for the variables of the function to minimize.
            - args (tuple): Other arguments for the function. 
    
        Returns:
            - array: An array containing the fidelity with vacuum.
            - array: An array containing the optimized gate parameters.
    
        Note:
        - The gate parameters are initialized with random values between 0 and 0.1.
        - The BFGS method is employed for optimization with a specified tolerance.
        """
        def optimize_circuit(): 
            result = scipy.optimize.minimize(self.cooled_squared_vac_overlap, 0.1*np.random.rand(self.num_blocks*len(self.gates)), args=args, method="BFGS", tol=1e-7)
            return -result['fun'], result['x'] 

        # Perform parallel optimization trials
        FidsSols  = Parallel(n_jobs=self.n_jobs)(delayed(optimize_circuit)() for trial in range(self.num_trials))
        self.Fids = np.array([FidSol[0] for FidSol in FidsSols]) # Fidelity
        self.Sols = np.array([FidSol[1] for FidSol in FidsSols]) # Gate parameters

        # Determine optimal parameter
        self.bestFid = np.max(self.Fids) # Highest fidelity
        self.bestSol = self.Sols[np.argmax(self.Fids)]
            
        return np.array([FidSol[0] for FidSol in FidsSols]), np.array([FidSol[1] for FidSol in FidsSols]) 


# 