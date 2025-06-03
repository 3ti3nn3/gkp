# import packages
import strawberryfields as sf 
import numpy as np


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


def prepare_state(rev_gates, rev_params, cutoff):
	'''
	Frontend funcion 

	Parameters
		- rev_gates (array: sf.ops): an array of gates from the strawberry operations library 
		- rev_params (array): an array of real numbers representing the parameters for the gates of the circuit
		- cutoff (integer): integer to describe the truncation of the Fock space
	Returns
		- BaseFockState: the output quantum state in the Fock basis
	'''
	# Check input 
	if int(len(rev_params)%len(rev_gates)) != 0: 
			raise ValueError("Params and gates don't match.")

	# Initialize Strawberry Fields program
	progParamState = sf.Program(1) # Photonic circuit with one mode

	with progParamState.context as q: 
		sf.ops.Vacuum() | q # Initialize vacuum

		for i in range(len(rev_params)//len(rev_gates)):
			for j in range(len(rev_gates)): 
				rev_gates[j](-rev_params[i*len(rev_gates)+j]) | q 

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

