import numpy as np 
import strawberryfields as sf 
from collections import Counter


# gates
def Scgate(phi): return sf.ops.Sgate(1, phi)
def Dcgate(phi): return sf.ops.Dgate(1, phi)
def Vconst(theta): return sf.ops.Vgate(np.sign(theta)*1)
def Vtanhgate(s): return sf.ops.Vgate(np.tanh(s))
def V4tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 4)
def V5gate(s): return sf.ops.Vkgate(s, 5)
def V5tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 5)
def V6tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 6)
def V7tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 7)
def V8tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 8)
def V9tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 9)
def V11tanhgate(s): return sf.ops.Vkgate(np.tanh(s), 11)
def Kdl2c2gate(kappa): return sf.ops.Kdlcgate(kappa, 2, 2)
def Kdl6c2gate(kappa): return sf.ops.Kdlcgate(kappa, 6, 2)
def Kdl10c2gate(kappa): return sf.ops.Kdlcgate(kappa, 10, 2)


# dictionaries
state_dict = {(0, 0): '0', (np.pi, 0): '1', (np.pi/2, 0): '+'} # dictionary to save the state properly
dict_gate = {
        'X': sf.ops.Xgate,
        'S': sf.ops.Sgate,
        'Sc': Scgate,
        'D': sf.ops.Dgate,
        'Dc': Dcgate, 
        'R': sf.ops.Rgate,
        'K': sf.ops.Kgate,
        'V': sf.ops.Vgate,
        'V5': V5gate,
        'Vtanh': Vtanhgate,
        'Vconst': Vconst,
        'V4tanh': V4tanhgate,
        'V5tanh': V5tanhgate, 
        'V6tanh': V6tanhgate,
        'V7tanh': V7tanhgate,
        'V8tanh': V8tanhgate, 
        'V9tanh': V9tanhgate,
        'V11tanh': V11tanhgate,
    }


# hyperparameters
backend = "fock"


# content 
def prepare_GKP(epsilon, cutoff, state):
    '''
    Prepare a GKP states with epsilon gaussian envolope. 

    Parameters: 
        - epsilon (float): finite energy parameter of the state
        - state (list): [theta,phi] for qubit definition
    Returns: 
        - BaseFockState: coefficients of the GKP state in fock basis
    '''
    progTarget = sf.Program(1)

    with progTarget.context as q: 
        # Initialize target state
        sf.ops.GKP(epsilon=epsilon, state=state) | q 
        
    # initialize engine, selecting Fock backend    
    eng = sf.Engine(backend, backend_options={"cutoff_dim": cutoff}) 
    
    return eng.run(progTarget, shots=1).state


def prepare_state(gates, params, cutoff):
    '''
    Frontend funcion 

    Parameters
        - gates (array: sf.ops): an array of gates from the strawberry operations library 
        - params (array): an array of real numbers representing the parameters for the gates of the circuit
        - cutoff (integer): integer to describe the truncation of the Fock space
    Returns
        - BaseFockState: the output quantum state in the Fock basis
    '''
    # Check input 
    if int(len(params)%len(gates)) != 0: 
            raise ValueError("Params and gates don't match.")

    # Initialize Strawberry Fields program
    progParamState = sf.Program(1) # Photonic circuit with one mode

    with progParamState.context as q: 
        sf.ops.Vacuum() | q # Initialize vacuum

        for i in range(len(params)//len(gates)):
            for j in range(len(gates)): 
                gates[j](params[i*len(gates)+j]) | q 

    # Initialize engine, selecting Fock backend
    eng = sf.Engine(backend, backend_options={"cutoff_dim": cutoff})

    # Output state of parametrized circuit
    return eng.run(progParamState,shots=1).state 


def wavefunction(q, ket):
    """
    Compute the wavefunction in position space from the quantum state vector in the Fock basis.

    Parameters:
        - q (array): An array of position values.
        - ket (array): Fock-basis representation of the quantum state.

    Returns:
        - array: Wavefunction in position basis.
    """
    c            = ket.shape[0]
    coefficients = np.zeros(c, dtype=complex)

    # Calculating the coefficients (that depend on n) of each Hermite polynomial i.e. coefficients[n] = <n|Psi_g>(n! 2**n)**(-1/2)
    coefficients[0] = 1
    for n in range(1,c):
        coefficients[n] = 1/(2*n) * coefficients[n-1]
    coefficients = np.sqrt(coefficients)*ket

    # Calculating the wavefunction, Psi(q):
    return np.exp(-q**2/2) * np.pi**(-1/4) * np.polynomial.hermite.hermval(q, coefficients)


def error_gancy_knill(ket_state, n=30, size=30):
    '''
    Calculating the probability of no uncorrectable errors, 
    P(no error) from Glancy & Knill (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.73.012325) 
    using Eq.(71) from Tzitrin et al. (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.101.032315). 

    Parameters: 
        - n (integer): Number of strips in integration
        - size (integer): Infinite sums over  s and t are truncated to go from -size to +size
        - state (array): state in Fock-basis
    Returns: 
        - float: P(no error)
    '''
    # Initialize matrix to store values of P_{no error} for different values of s and t, before we take the sum
    beforesum = np.zeros((2*size,2*size), dtype=complex) 
        
    for t in range(-size,size):
        for s in range(-size,size):
            x = np.linspace((np.sqrt(np.pi))*(2*s-1/6), (np.sqrt(np.pi))*(2*s+1/6), n) # Position axis
            
            # Filling the matrix
            beforesum[t+size,s+size] = (1/3) * np.sinc(t/3) * np.trapz(np.conj(wavefunction(x+2*t*np.sqrt(np.pi),ket_state))*wavefunction(x,ket_state), x)
                
    return 1.-np.abs(np.sum(beforesum))


def string_to_gateset(str_gate): 
    '''
    Takes a string of gates and transforms to a list of the corresponding gates.
    dict_gate is necessary.

    Parameters: 
        str_gates (str): gates as string
    Results: 
        (list): list of gates 
    '''
    # Sort keys by length descending so longer matches are attempted first
    keys = sorted(dict_gate.keys(), key=len, reverse=True)
    
    result = []
    i = 0
    while i < len(str_gate):
        for key in keys:
            if str_gate.startswith(key, i):
                result.append(dict_gate[key])
                i += len(key)
                break
        else:
            raise ValueError(f"Unknown token at position {i} in string: {str_gate[i:]}")
    
    # Print the result as a list of arrays
    return result


def count_gates(str_gate):
    '''
    Counts the number of gates in a string. 

    Parameters: 
        str_gate (str): string of gates
    Results: 
        (dict): dict[<gate>] = <gate count> 
    '''
    # Sort keys by length descending so longer matches are attempted first
    keys = sorted(dict_gate.keys(), key=len, reverse=True)
    
    count = 0
    i = 0
    while i < len(str_gate):
        for key in keys:
            if str_gate.startswith(key, i):
                count += 1
                i += len(key)
                break
        else:
            raise ValueError(f"Unknown token at position {i} in string: {str_gate[i:]}")
    
    # Print the result as a list of arrays
    return count


def gate_ratio(str_gate):
    '''
    Computes the relative ratios of gates in a string. 

    Parameters: 
        str_gate (str): string of gates
    Results: 
        (dict): dict[<gate>] = <relative gate count> 
    '''
    # Sort keys by length descending so longer matches are attempted first
    keys = sorted(dict_gate.keys(), key=len, reverse=True)
    
    gate_list = []
    i = 0
    while i < len(str_gate):
        for key in keys:
            if str_gate.startswith(key, i):
                gate_list.append(key)
                i += len(key)
                break
        else:
            raise ValueError(f"Unknown token at position {i} in string: {str_gate[i:]}")
    
    total = len(gate_list)
    counts = Counter(gate_list)
    ratios = {gate: count / total for gate, count in counts.items()}
    
    return ratios


def draw_random_without_double(n, elements, weights):
    '''
    Samples a string of gates from the elements list according to the weights.
    It applies the rule that not the same elements can be consecutive. 

    Parameters:
        n (int): length of the sample
        elements (list): list of elements to sample from
        weights (list): list according to which one samples 
    Results:
        str: string of gates
    '''
    result = ''
    last_drawn = None

    for _ in range(n):
        drawn_element = draw_random_element(elements, last_drawn, weights)
        result += drawn_element
        last_drawn = drawn_element

    return result


def draw_random_element(elements, last_drawn, weights):
    '''
    Samples one element of elements while avoiding to draw the same element as last_drawn.

    Parameters: 
        elements (list): list of elements to sample from
        last_drawn (str): element which was drawn before 
        weights (list): weights according to which one element of elements will sampled from 
    Results: 
        str: sampled element
    '''
    if last_drawn is not None:
        # Create a list of available elements excluding the last drawn element
        available_elements = [e for e in elements if e != last_drawn]
        # Filter the weights accordingly
        available_weights = [w for e, w in zip(elements, weights) if e != last_drawn]
    else:
        available_elements = elements
        available_weights = weights

    # Draw a random element from the available elements with the corresponding weights
    drawn_element = random.choices(available_elements, weights=available_weights, k=1)[0]

    return drawn_element