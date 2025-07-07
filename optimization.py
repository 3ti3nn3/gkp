# import packages
import numpy as np
import strawberryfields as sf
import scipy
from joblib import Parallel, delayed
import pickle
import os
import auxiliary
import time


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

    # Optimization function 
    circuit = ''

    # Optimization results
    Fids    = None
    Sols    = None
    bestFid = None
    bestSol = None

    # Gates
    gates = None 

    # Optimization time 
    time = None

    
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
            self.time    = None

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


    # access time
    def get_time(self):
        return self.time


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
    def parallel_optimize_circuit(self, params=None, args=()):
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
        if params is None: 
            def optimize_circuit(): 
                result = scipy.optimize.minimize(self.cooled_squared_vac_overlap, 0.1*np.random.rand(self.num_blocks*len(self.gates)), args=args, method="BFGS", tol=1e-7)
                return -result['fun'], result['x'] 
        else:
            def optimize_circuit(): 
                result = scipy.optimize.minimize(self.cooled_squared_vac_overlap, params, args=args, method="BFGS", tol=1e-7)
                return -result['fun'], result['x'] 

        # Perform parallel optimization trials
        time1 = time.clock_gettime(0)
        FidsSols  = Parallel(n_jobs=self.n_jobs)(delayed(optimize_circuit)() for trial in range(self.num_trials))
        self.time = time.clock_gettime(0) - time1

        self.Fids = np.array([FidSol[0] for FidSol in FidsSols]) # Fidelity
        self.Sols = np.array([FidSol[1] for FidSol in FidsSols]) # Gate parameters

        # Determine optimal parameter
        self.bestFid = np.max(self.Fids) # Highest fidelity
        self.bestSol = self.Sols[np.argmax(self.Fids)]
            
        return np.array([FidSol[0] for FidSol in FidsSols]), np.array([FidSol[1] for FidSol in FidsSols])


    # rebuild state from best fidelity parameters
    def prepare_best_state(self, gates_rev=None, params_rev=None):
        '''
        Parameters
            - gates_rev (list): list of gates in reversed order for the case where the 
                reverse order of the gates is not equal to the gates needed to rebuild 
                the GKP state
            - params_rev (list): list of the parameters to rebuild the GKP state
        Returns
            - BaseFockState: the output quantum state in the Fock basis
        '''
        # Check whether there is some other data transfered
        if gates_rev is None and params_rev is None:
            gates_rev = np.flip(self.gates) 
            params_rev = - np.flip(self.bestSol)

        # Initialize Strawberry Fields program
        progParamState = sf.Program(1) # Photonic circuit with one mode

        # consistency check 
        if len(self.bestSol)//len(self.gates) != self.num_blocks:
            raise ValueError('There is an inconsistency in the number of blocks and the number of parameters.')

        with progParamState.context as q: 
            sf.ops.Vacuum() | q # Initialize vacuum

            for i in range(self.num_blocks):
                for j in range(len(gates_rev)): 
                    gates_rev[j](params_rev[i*len(gates_rev)+j]) | q 

        # Initialize engine, selecting Fock backend
        eng = sf.Engine(backend, backend_options={"cutoff_dim": self.cutoff})

        # Output state of parametrized circuit
        return eng.run(progParamState,shots=1).state


    # compute the norm of the rebuild state
    def norm_best_state(self):
        '''
        Compute the norm of the rebuild states. 

        Parameters
        Returns
            - float: norm of the generated state according to the best fidelity parameters
        '''
        ket_best = self.prepare_best_state().ket()
        ret = np.abs(np.dot(ket_best.conj(), ket_best))

        return ret


    # check consistency of rebuild method 
    def check_fidelity(self):
        '''
        Check whether the fidelity between the target state and the reconstructed state is the same as the best fidelity.

        Parameters
        Return
            - boolean: indicating the consistency or not 
        '''
        # prepare GKP state
        progTarget = sf.Program(1)

        with progTarget.context as q: 
            # Initialize target state
            sf.ops.GKP(epsilon=self.epsilon, state=self.state) | q 
            
        # initialize engine, selecting Fock backend    
        eng = sf.Engine("fock", backend_options={"cutoff_dim": self.cutoff}) 
        
        ket_GKP = eng.run(progTarget, shots=1).state.ket()
        ket_best = self.prepare_best_state().ket()

        fid = np.abs(np.dot(ket_GKP.conj(), ket_best))**2

        ret = np.abs(fid - self.bestFid) < 1e-5

        return ret


# class for pretrained circuits 
class OptimizationPre(Optimization): 
    '''
    Class takes extra list of sets with already trained data and then trains the training
    parameters according 
    '''
    cutoff     = 10
    delta      = 0.5 
    epsilon    = 0.5**2 
    num_blocks = 6 
    num_trials = 2 
    n_jobs     = 2 
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
    gatesPre = None
    paramsPre = None
    
    # initializer
    def __init__(self, gates=None, gatesPre=None, paramsPre=None, circuit=None, cutoff=None, delta=None, epsilon=None, num_blocks=None, num_trials=None, n_jobs=None, state=None, file_name=None): 
        
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

            self.gatesPre = gatesPre
            self.paramsPre = paramsPre

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

            # Use pre-trained data
            for i in range(len(self.paramsPre)//len(self.gatesPre)):
                for k in range(len(self.gatesPre)): 
                    self.gatesPre[k](self.paramsPre[i*len(self.gatesPre)+k]) | q

            # optimization step
            for i in range(len(params)//len(self.gates)):
                for k in range(len(self.gates)): 
                    self.gates[k](params[i*len(self.gates)+k]) | q

        # Initialize engine, selecting Fock backend (cutoff is hyperparameter of the file)
        eng = sf.Engine(backend, backend_options={"cutoff_dim": self.cutoff}) 
    
        # Execute program on engine
        cooled_state = eng.run(prog, shots=1).state # output state of parametrized circuit
        
        # Return fidelity with vacuum (negative for minimization later)
        return -cooled_state.fock_prob([0])


    # rebuild state from best fidelity parameters
    def prepare_best_state(self, gates_rev=None, params_rev=None):
        '''
        Parameters
            - gates_rev (list): list of gates in reversed order for the case where the 
                reverse order of the gates is not equal to the gates needed to rebuild 
                the GKP state
            - params_rev (list): list of the parameters to rebuild the GKP state
        Returns
            - BaseFockState: the output quantum state in the Fock basis
        '''
        # Check whether there is some other data transfered
        if gates_rev is None and params_rev is None:
            gates_rev = np.flip(self.gates) 
            params_rev = - np.flip(self.bestSol)

        gatesPre_rev = np.flip(self.gatesPre)
        paramsPre_rev = - np.flip(self.paramsPre)

        # Initialize Strawberry Fields program
        progParamState = sf.Program(1) # Photonic circuit with one mode

        # consistency check 
        if len(self.bestSol)//len(self.gates) != self.num_blocks:
            raise ValueError('There is an inconsistency in the number of blocks and the number of parameters.')


        with progParamState.context as q: 
            sf.ops.Vacuum() | q # Initialize vacuum

            for i in range(self.num_blocks):
                for j in range(len(gates_rev)): 
                    gates_rev[j](params_rev[i*len(gates_rev)+j]) | q 

            # Use pre-trained data
            for i in range(len(paramsPre_rev)//len(gatesPre_rev)):
                for k in range(len(gatesPre_rev)): 
                    gatesPre_rev[k](paramsPre_rev[i*len(gatesPre_rev)+k]) | q

        # Initialize engine, selecting Fock backend
        eng = sf.Engine(backend, backend_options={"cutoff_dim": self.cutoff})

        # Output state of parametrized circuit
        return eng.run(progParamState,shots=1).state


# content
class OptimizationBlock(Optimization): 

    cutoff     = 10
    delta      = 0.5
    epsilon    = 0.5**2
    num_blocks = 6
    num_trials = 2
    n_jobs     = 2 
    state      = [0,0]

    # Optimisation function 
    circuit = ''

    # Optimization results
    Fids    = None
    Sols    = None
    bestFid = None
    bestSol = None

    # Gates and pretrained parameters
    gateBlocks = None
    paramBlocks = None
    
    # initializer
    def __init__(self, gateBlocks=None, paramBlocks=None, circuit=None, cutoff=None, delta=None, epsilon=None, num_blocks=None, num_trials=None, n_jobs=None, state=None, file_name=None): 
        
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

            self.gateBlocks = gateBlocks
            self.paramBlocks = paramBlocks
            self.circuit = circuit

        # if the date is loaded from a file
        elif type(file_name) == str: 
            with open('data/' + file_name, 'rb') as file:
                saved_dict = pickle.load(file)
            self.__dict__.update(saved_dict)

        # cause error
        else: 
            raise ValueError(f'"{file_name}" is an invalid file name!')


    # access paramBlock
    def get_paramBlock(): 
        return self.paramBlock


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

            for i in range(len(params)//len(self.gateBlocks)):
                for j in range(len(self.gateBlocks)): 
                    for k in range(len(self.paramBlocks[j])//len(self.gateBlocks[j])):
                        for l in range(len(self.gateBlocks[j])): 
                            self.gateBlocks[j][l](params[i*len(self.gateBlocks)+j] * self.paramBlocks[j][k*len(self.gateBlocks[j])+l]) | q


        # Initialize engine, selecting Fock backend (cutoff is hyperparameter of the file)
        eng = sf.Engine(backend, backend_options={"cutoff_dim": self.cutoff}) 
    
        # Execute program on engine
        cooled_state = eng.run(prog, shots=1).state # output state of parametrized circuit
        
        # Return fidelity with vacuum (negative for minimization later)
        return -cooled_state.fock_prob([0])


    # parallel trials for optimizing the gate parameters to maximize fidelity
    def parallel_optimize_circuit(self, params=None, args=()):
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
        if params is None: 
            def optimize_circuit(): 
                result = scipy.optimize.minimize(self.cooled_squared_vac_overlap, 0.1*np.random.rand(self.num_blocks*len(self.gateBlocks)), args=args, method="BFGS", tol=1e-7)
                return -result['fun'], result['x'] 
        else:
            def optimize_circuit(): 
                result = scipy.optimize.minimize(self.cooled_squared_vac_overlap, params, args=args, method="BFGS", tol=1e-7)
                return -result['fun'], result['x'] 

        # Perform parallel optimization trials
        FidsSols  = Parallel(n_jobs=self.n_jobs)(delayed(optimize_circuit)() for trial in range(self.num_trials))
        self.Fids = np.array([FidSol[0] for FidSol in FidsSols]) # Fidelity
        self.Sols = np.array([FidSol[1] for FidSol in FidsSols]) # Gate parameters

        # Determine optimal parameter
        self.bestFid = np.max(self.Fids) # Highest fidelity
        self.bestSol = self.Sols[np.argmax(self.Fids)]
            
        return np.array([FidSol[0] for FidSol in FidsSols]), np.array([FidSol[1] for FidSol in FidsSols]) 
