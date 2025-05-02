import numpy as np
from numpy import linalg as la
from itertools import permutations

#%% Global variables
Identity2 = np.array([[1, 0], [0, 1]])
sigma1 = np.array([[0, 1], [1, 0]])
sigma2 = np.array([[0, -1j], [1j, 0]])
sigma3 = np.array([[1, 0], [0, -1]])
sigma = [sigma1, sigma2, sigma3]

Identity3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
Lambda1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Lambda2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
Lambda3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
Lambda4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
Lambda5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
Lambda6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Lambda7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
Lambda8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]])/np.sqrt(3)
Lambda = [Lambda1, Lambda2, Lambda3, Lambda4, Lambda5, Lambda6, Lambda7, Lambda8]


#%%#############################Functions###############################
def square_matrix(line_matrix: list[complex], epsilon = 1e-5) -> list[complex]:
        """
        This function converts a hermitian matrix expressed as a line into a typical matrix form.
        Caution : the definition of matrices in fortran and python are not the same.
        """
        length_lin = len(line_matrix)
        n = (-1 + np.sqrt(1 + 8*length_lin))/2
        if n%1 != 0:
                return "Error :The line matrix given has incorrect length"
        else:
                n = int(n)

        matrix_square = np.zeros((n, n), dtype=np.complex_)

        for i in range(n):
                for k in range(n):
                        if k > i:
                                matrix_square[i][k] = np.conjugate(line_matrix[i*n + k - i*(i + 1)//2]) 
                        elif k == i:
                                matrix_square[i][k] = line_matrix[i*n + k - i*(i + 1)//2] #this line is just here because the diagonal elements should not be conjugated (they are real anyway)
                        else:
                                matrix_square[i][k] = np.conjugate(matrix_square[k][i])
                                                                
        return matrix_square

#%% from line first to colum first
def invert_momenta(p:list[float]) ->list[float]:
        """
        fortran/C-python do not order table in the same order, one should use this function on any momentum
        from fortran that is used in python functions
        """
        new_p = []
        for i in range(len(p[0])):  new_p.append([0]*len(p))
        for i, onep in enumerate(p):
            for j, x in enumerate(onep):
                new_p[j][i] = x
        return new_p

#%% It should work but need to check twice
def write_density_matrix(rho: list[complex], pos: list[int], allow_hel: list[int], epsilon=1e-10) -> None:
        """
        Writes the density matrix terms and the helicities of each amplitude used for the calculation of each term
        """
        n_changing = len(pos)
        if n_changing == 0:
               return "Error :All helicities are fixed, cannot compute density matrix"
        
        n_comb = int(len(allow_hel)/n_changing)
        sol = 0
        for i in range(n_comb):
                for j in range(i, n_comb):
                        sol += 1
                        for k in range(n_changing):
                                print('particle', pos[k], 'has helicity', allow_hel[i*n_changing + k], allow_hel[j*n_changing + k])

                        #We put a cut here to make the output more readable
                        if np.abs(rho[sol - 1]) < epsilon:
                                rho[sol - 1] = 0.

                        print('value is', sol, rho[sol - 1])
        return

#%%ok for n
def get_trace(line_matrix: list[complex], n_comb: int) -> float:
        """
        Function that computes the trace of a line matrix for a sliced line matrix
        """
        diagIndices = [0] * n_comb
        counter = 0
        for idiag in range(n_comb):
               diagIndices[idiag] = idiag * n_comb - counter
               counter += idiag
        #print("DiagIndices :", diagIndices)
        trace = 0
        for i in range(n_comb):
               trace += np.real(line_matrix[diagIndices[i]])

        return trace

def get_rho_normalised(rho: list[complex], n_comb: int, epsilon=1e-10) -> list[complex]:
        """
        This simple function takes a line density matrix rho, computes its trace
        and normalises rho by its trace
        """
        trace = get_trace(rho, n_comb)
        #Put a threshold, ça ne marche pas pour l'instant
        for i in range(len(rho)):
                if abs(np.real(rho[i]))/trace < epsilon:
                        rho[i] = rho[i].imag
                elif abs(np.imag(rho[i]))/trace < epsilon:
                        rho[i] = rho[i].real
        aux = np.array(rho) / trace
        return aux

#%%Very simple function, just to make code nicer
def get_list_sliced(list: list[complex], n_comb: int, epsilon = 1e-10) -> list[complex]:
        """
        The fortran code defines list of length 99. This function removes 
        the elements that are not relevant for a more practical use. 
        Also we put a threshold on the values to eliminate numerrical errors
        """
        if isinstance(n_comb, int) :
                list = list[0: int(n_comb*(n_comb + 1)/2)]
                trace = get_trace(list, n_comb)
                for i in range(len(list)):
                        if abs(np.real(list[i]))/trace < epsilon:
                                list[i] = list[i].imag
                        elif abs(np.imag(list[i]))/trace < epsilon:
                                list[i] = list[i].real
                return list
        else:
                return "Error: n_comb is not an integer"
       
#%%This function computes the spin correlation matrix for square density matrix
def Get_Correlations(rho:list[complex], pdg_pos:list[int], epsilon=1e-10) -> list[float, float]:
        """
        This function takes as input a density matrix in a square (python) format. 
        It then calculates the correlation matrix of the spin of the two 
        fermions/two vector bosons with the trace formula.
        Update it the same way as the polarisations algorithm
        """
        fermions_pdg = [1, 2, 3, 4, 5, 6, 7, 8] + [-1, -2, -3, -4, -5, -6, -7, -8] + [11, 12, 13, 14, 15, 16, 17, 18] + [-11, -12, -13, -14, -15, -16, -17, -18]
        boson_pdg = [9, 21, 22, 23, 24, -24] #these 3 entries could be put as inputs to generalise to any model, for instance with Higgs characterozation one
        scalar_pdg = [25, 37]
        if pdg_pos[0] in fermions_pdg: 
                if pdg_pos[1] in fermions_pdg: #both particles are fermion
                        C = np.zeros((3, 3), dtype=np.complex128)
                        for i in range(3):
                                for j in range(3):
                                        tensorprod = np.kron(sigma[i], sigma[j])
                                        C[i][j] = np.real(np.trace(np.dot(tensorprod, rho)))
                                        if np.abs(C[i][j]) < epsilon:
                                                C[i][j] = 0.
                elif pdg_pos[1] in boson_pdg: #1st particle fermion, 2nd massive boson
                        C = np.zeros((3, 8), dtype=np.complex128)
                        for i in range(3):
                                for j in range(8):
                                        tensorprod = np.kron(sigma[i], Lambda[j])
                                        C[i][j] = np.real(np.trace(np.dot(tensorprod, rho))/2.)
                                        if np.abs(C[i][j]) < epsilon:
                                                C[i][j] = 0.
                elif pdg_pos[1] in scalar_pdg: #1st particle fermion, 2nd scalar
                        raise ValueError('There is no spin-correlation with a scalar')
                else:
                        raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')

        elif pdg_pos[0] in boson_pdg:
                if pdg_pos[1] in fermions_pdg: #1st particle massive boson, 2nd fermion
                       C = np.zeros((8, 3), dtype=np.complex128)
                       for i in range(8):
                                for j in range(3):
                                        tensorprod = np.kron(Lambda[i], sigma[j])
                                        C[i][j] = np.real(np.trace(np.dot(tensorprod, rho))/2.)
                                        if np.abs(C[i][j]) < epsilon:
                                                C[i][j] = 0.
                elif pdg_pos[1] in boson_pdg: #both particles are massive boson
                        C = np.zeros((8, 8), dtype=np.complex128)
                        for i in range(8):
                                for j in range(8):
                                        tensorprod = np.kron(Lambda[i], Lambda[j]) #the tensor product is with Gell-Mann matrices
                                        C[i][j] = np.real(np.trace(np.dot(tensorprod, rho))/4.) #there is an additional 1/4 factor
                                        if np.abs(C[i][j]) < epsilon:
                                                C[i][j] = 0.
                elif pdg_pos[1] in scalar_pdg: #1st particle massive boson, 2nd scalar
                        raise ValueError('There is no spin-correlation with a scalar')
                else:
                        raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')              
        elif pdg_pos[0] in scalar_pdg:
               raise ValueError('There is no spin-correlation with a scalar')
        else:
                raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')     

        C2 = np.dot(np.transpose(C),C)

        return C, C2

#%%This functon computes the polarisation of the qubits for square density matrix
def Get_Polarisations(rho:list[complex], pdg_pos:list[int], epsilon=1e-10) -> list[float]:
        """
        This function takes as input a density matrix in a square (python) format. 
        It then calculates the polarisations of the two fermions/two vector bosons
        using the trace formula.
        Works only for 2 particles for now
        """
        fermions_pdg = [1, 2, 3, 4, 5, 6, 7, 8] + [-1, -2, -3, -4, -5, -6, -7, -8] + [11, 12, 13, 14, 15, 16, 17, 18] + [-11, -12, -13, -14, -15, -16, -17, -18]
        boson_pdg = [9, 21, 22, 23, 24, -24] #these 3 entries could be put as inputs to generalise to any model, for instance with Higgs characterozation one
        scalar_pdg = [25, 37]
        #pdg_pos[0] is the first particle
        if pdg_pos[0] in fermions_pdg: #the first particle is a fermion or a massless vector boson -> it has 2 spin states possibles
                B1 = np.zeros(3, dtype=np.complex128)
                if pdg_pos[1] in fermions_pdg: #1st particle fermion, 2nd fermion
                        for i in range(3):
                                tensorprod1 = np.kron(sigma[i], Identity2)
                                B1[i] = np.trace(np.dot(tensorprod1, rho))
                                if np.abs(B1[i].real) < epsilon:
                                        B1[i] = B1[i].imag*1j
                                if np.abs(B1[i].imag) < epsilon:
                                        B1[i] = B1[i].real
                elif pdg_pos[1] in boson_pdg == 24: #1st particle fermion, 2nd massive vector boson
                        for i in range(3):
                                tensorprod1 = np.kron(sigma[i], Identity3)
                                B1[i] = np.trace(np.dot(tensorprod1, rho))/2.
                                if np.abs(B1[i].real) < epsilon:
                                        B1[i] = B1[i].imag*1j
                                if np.abs(B1[i].imag) < epsilon:
                                        B1[i] = B1[i].real
                else:
                        raise ValueError("One of the particle chosen is either a scalar or not recognised")

        elif pdg_pos[0] in boson_pdg: #massive vector boson -> it has 3 spin states possibles
                B1 = np.zeros(8, dtype=np.complex128)
                if pdg_pos[1] in fermions_pdg: #1st particle massive vector boson, 2nd fermion
                        for i in range(8):
                                tensorprod1 = np.kron(Lambda[i], Identity2)
                                B1[i] = np.trace(np.dot(tensorprod1, rho))/2.
                                if np.abs(B1[i].real) < epsilon:
                                        B1[i] = B1[i].imag*1j
                                if np.abs(B1[i].imag) < epsilon:
                                        B1[i] = B1[i].real

                elif pdg_pos[1] in boson_pdg: #1st particle massive vector boson, 2nd massive vector boson
                        for i in range(8):
                                tensorprod1 = np.kron(Lambda[i], Identity3)
                                B1[i] = np.trace(np.dot(tensorprod1, rho))/2.
                                if np.abs(B1[i].real) < epsilon:
                                        B1[i] = B1[i].imag*1j
                                if np.abs(B1[i].imag) < epsilon:
                                        B1[i] = B1[i].real
                else:
                        raise ValueError("One of the particle chosen is either a scalar or not recognised")
        elif pdg_pos[0] in scalar_pdg: #scalar -> it has 1 spin state possible
                B1 = [1. + 0.*1j]
        else:
                raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')


        #pdg_pos[1] is the second particle
        if pdg_pos[1] in fermions_pdg: #the second particle is a fermion or a massless vector boson -> it has 2 spin states possibles
                B2 = np.zeros(3, dtype=np.complex128)
                if pdg_pos[0] in fermions_pdg: #both particles are fermions
                        for i in range(3):
                                tensorprod2 = np.kron(Identity2, sigma[i])
                                B2[i] = np.trace(np.dot(tensorprod2, rho))
                                if np.abs(B2[i].real) < epsilon:
                                        B2[i] = B2[i].imag*1j
                                if np.abs(B2[i].imag) < epsilon:
                                        B2[i] = B2[i].real
                elif pdg_pos[0] in boson_pdg: #first particle massive vector boson, second fermion
                        for i in range(3):
                                tensorprod2 = np.kron(Identity3, sigma[i])
                                B2[i] = np.trace(np.dot(tensorprod2, rho))/2.
                                if np.abs(B2[i].real) < epsilon:
                                        B2[i] = B2[i].imag*1j
                                if np.abs(B2[i].imag) < epsilon:
                                        B2[i] = B2[i].real
                else:
                        raise ValueError("One of the particle chosen is either a scalar or not recognised")
                
        elif pdg_pos[1] in boson_pdg: #massive vector boson -> it has 3 spin states possibles
                B2 = np.zeros(8, dtype=np.complex128)
                if pdg_pos[0] in fermions_pdg: #first particle fermion second vector boson
                        for i in range(8):
                                tensorprod2 = np.kron(Identity2, Lambda[i])
                                B2[i] = np.trace(np.dot(tensorprod2, rho))/2.
                                if np.abs(B2[i].real) < epsilon:
                                        B2[i] = B2[i].imag*1j
                                if np.abs(B2[i].imag) < epsilon:
                                        B2[i] = B2[i].real 
                elif pdg_pos[0] in boson_pdg: #both particle massive vector boson
                        for i in range(8):
                                tensorprod2 = np.kron(Identity3, Lambda[i])
                                B2[i] = np.trace(np.dot(tensorprod2, rho))/2.
                                if np.abs(B2[i].real) < epsilon:
                                        B2[i] = B2[i].imag*1j
                                if np.abs(B2[i].imag) < epsilon:
                                        B2[i] = B2[i].real 
        elif pdg_pos[1] in scalar_pdg: #scalar -> it has 1 spin state possible
                B2 = [1. + 0.*1j]
        else:
                raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')
        
        return B1, B2


def spin_expectation(Polarisation: list[float]) -> list[float]:
        if len(Polarisation) == 8:
                spin_exp = [0., 0., 0.]
                spin_exp[0] = np.sqrt(2) * (Polarisation[0] + Polarisation[5])
                spin_exp[1] = np.sqrt(2) * (Polarisation[1] + Polarisation[6])
                spin_exp[2] = Polarisation[2] + np.sqrt(3) * Polarisation[7]
        
        return np.real(spin_exp)

def spinspin_expectation(Correlation: list[float]) -> list[float] | str:
        if len(Correlation) == 8:
                spin_exp = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]

                spin_exp[0][0] = 2 * (Correlation[0][0] + Correlation[0][5] + Correlation[5][0] + Correlation[5][5])
                spin_exp[0][1] = 2 * (Correlation[0][1] + Correlation[0][6] + Correlation[5][1] + Correlation[5][6])
                spin_exp[0][2] = np.sqrt(2) * (Correlation[0][2] + Correlation[5][2] + np.sqrt(3) * (Correlation[0][7] + Correlation[5][7]))

                spin_exp[1][0] = 2 * (Correlation[1][0] + Correlation[1][5] + Correlation[6][0] + Correlation[6][5])
                spin_exp[1][1] = 2 * (Correlation[1][1] + Correlation[1][6] + Correlation[6][1] + Correlation[6][6])
                spin_exp[1][2] = np.sqrt(2) * (Correlation[1][2] + Correlation[6][2] + np.sqrt(3) * (Correlation[1][7] + Correlation[6][7]))

                spin_exp[2][0] = np.sqrt(2) * (Correlation[2][0] + Correlation[2][5] + np.sqrt(3) * (Correlation[7][0] + Correlation[7][5]))
                spin_exp[2][1] = np.sqrt(2) * (Correlation[2][1] + Correlation[2][6] + np.sqrt(3) * (Correlation[7][1] + Correlation[7][6]))
                spin_exp[2][2] = Correlation[2][2] + 3 * Correlation[7][7] + np.sqrt(3) * (Correlation[2][7] + Correlation[7][2])
        else:
                return "Spin-spin correlation not implemented for anything else than qutrit/qutrit"
                
        return np.real(spin_exp)

def Partial_Trace(rho:list[complex, complex], index, pdg_pos:list[int]) -> list[complex, complex]:
    """
    Calculates the partial trace of a density matrix of a pair of qubits or qutrits or a system qubit/qutrit
    index = 1, tracing over the first Hilbert Space
    index = 2, tracing over the second Hilbert space
    """
    if rho.shape[0] != rho.shape[1]:
        raise ValueError('Asked the partial trace of a non-square matrix, problem')
    
    if np.sqrt(rho.shape[0])%1 == 0.: #the density matrix is composed of two systems of same dimension
        n = int(np.sqrt(rho.shape[0]))
        Basis = np.eye(n, dtype=int)
        TracedRho = 0
        if index == 1:
                for i in range(n):
                        Term1 = np.kron(Basis[i], Basis)
                        Term2 = np.transpose(np.kron(Basis[i], Basis))
                        TracedRho += np.dot(Term1, np.dot(rho, Term2))
        elif index == 2:
                for i in range(n):
                        Term1 = np.kron(Basis, Basis[i])
                        Term2 = np.transpose(np.kron(Basis, Basis[i]))
                        TracedRho += np.dot(Term1, np.dot(rho, Term2))
        else:
                raise ValueError('The value of index is not correct')

#the density matrix is composed of two systems of different dimensions, from rho itself we can't know which one is the first particle so we need to propagate the pdg-id codes
    elif rho.shape[0] == 6: #this case is the one for fermion-boson or boson-fermion
        if abs(pdg_pos[0]) <= 22 and (abs(pdg_pos[1]) == 23 or abs(pdg_pos[1]) == 24): #fermion 1st particle
                if index == 1:
                        n = 2
                        Basis = np.eye(n, dtype=int)
                        TracedRho = 0
                        for i in range(n):
                                Term1 = np.kron(Basis[i], Identity3)
                                Term2 = np.transpose(np.kron(Basis[i], Identity3))
                                TracedRho += np.dot(Term1, np.dot(rho, Term2))

                elif index == 2:
                        n = 3
                        Basis = np.eye(n, dtype=int)
                        TracedRho = 0
                        for i in range(n):
                                Term1 = np.kron(Identity2, Basis[i])
                                Term2 = np.transpose(np.kron(Identity2, Basis[i]))
                                TracedRho += np.dot(Term1, np.dot(rho, Term2))
                else:
                        raise ValueError('The value of index is not correct')
                
        elif (abs(pdg_pos[0]) == 23 or abs(pdg_pos[0]) == 24) and abs(pdg_pos[1]) <= 22: #boson 1st particle
                if index == 1:
                        n = 3
                        Basis = np.eye(n, dtype=int)
                        TracedRho = 0
                        for i in range(n):
                                Term1 = np.kron(Basis[i], Identity2)
                                Term2 = np.transpose(np.kron(Basis[i], Identity2))
                                TracedRho += np.dot(Term1, np.dot(rho, Term2))

                elif index == 2:
                        n = 2
                        Basis = np.eye(n, dtype=int)
                        TracedRho = 0
                        for i in range(n):
                                Term1 = np.kron(Identity3, Basis[i])
                                Term2 = np.transpose(np.kron(Identity3, Basis[i]))
                                TracedRho += np.dot(Term1, np.dot(rho, Term2))
                else:
                        raise ValueError('The value of index is not correct')
    
    return TracedRho

def ConcLB2(Rho: list[complex, complex], pdg_pos= list[int])-> float:
    """
    Return the square of the lower bound of the concurrence for a system composed of a pair of qutrits.
    See 2307.09675v2.
    """

    RA = Partial_Trace(Rho, 2, pdg_pos)
    TrB = np.trace(RA)
    RB = Partial_Trace(Rho, 1, pdg_pos)
    TrA =  np.trace(RB)

    RhoA = RA/TrA
    RhoB = RB/TrB

    aux1 = np.trace(Rho**2) - np.trace(RhoA**2)
    aux2 = np.trace(Rho**2) - np.trace(RhoB**2)
    return 2 * max(0, aux1, aux2)

def ConcUB2(Rho: list[complex, complex], pdg_pos: list[int])-> float:
    """
    Return the square of the upper bound of the concurrence for a system composed of a pair of qutrits.
    See 2307.09675v2.
    """

    RA = Partial_Trace(Rho, 2, pdg_pos)
    TrB = np.trace(RA)
    RB = Partial_Trace(Rho, 1, pdg_pos)
    TrA =  np.trace(RB)

    RhoA = RA/TrA
    RhoB = RB/TrB

    aux1 = 1 - np.trace(RhoA**2)
    aux2 = 1 - np.trace(RhoB**2)
    return 2 * min(aux1, aux2)

#%% Rotation for p = p[E, px, py, pz]
#it must be applied on momenta in the python format (so before invert_momenta)
def rotation(p: list[float], phi: float, theta: float, epsilon: float) ->list[float]:
        """
        This function takes as input a momentum of a single particle. It rotates
        it with angles theta & phi. CAUTION: it takes the input in the python format (so before invert_momenta)
        """
        pRot = [0, 0, 0, 0]

        if np.abs(p[1]) < 1e-10 and np.abs(p[2]) < 1e-10 and np.abs(p[3]) < 1e-10: #if the particle is immobile
                pRot = [p[0], 0., 0., 0.]
        else:

                pRot[0] = p[0] #Energy
                pRot[1] = -np.sin(phi) * p[1] + np.cos(phi) * p[2] #p_n
                pRot[2] = -np.cos(phi) * np.cos(theta) * p[1] - np.sin(phi) * np.cos(theta) * p[2] + np.sin(theta) * p[3] #p_r
                pRot[3] = np.cos(phi) * np.sin(theta) * p[1] + np.sin(phi) * np.sin(theta) * p[2] + np.cos(theta) * p[3] #p_k

        #If the transversal momenta are very small (ratio to longitudinal momenta < epsilon ~ 10^-12) they are put to exactly zero because it changes polarisation signs.
                if np.abs(pRot[1]/pRot[3]) < epsilon:
                        pRot[1] = 0.

                if np.abs(pRot[2]/pRot[3]) < epsilon:
                        pRot[2] = 0.

        return pRot

def boost(p:list[float], q:float, m:float) -> list[float]:
        '''
        This function takes as argument the 4-momentum of a single particle and returns the 4-momentum boosted by q in an arbitrary direction.
        Inputs
        p(0:3): four-momentum p in the q rest  frame
        q(0:3): four-momentum q in the boosted frame
        m: mass of q (for numerical stability)
        '''

        pboost = [0] * 4
        qq = q[1]**2+q[2]**2+q[3]**2
        if qq > 0:
                pq = p[1]*q[1] + p[2]*q[2] + p[3]*q[3]
                lf = ((q[0]-m) * pq/qq + p[0])/m
                pboost[0] = (p[0]*q[0] + pq)/m
                pboost[1] = p[1] + q[1] * lf
                pboost[2] = p[2] + q[2] * lf
                pboost[3] = p[3] + q[3] * lf
        else:
                pboost = p

        return pboost

def cut_matrix(square_matrix: list[complex]) -> list[complex]:
        '''
        Removes from a square matrix the terms that are smaller than 1e-10 (or their norm is complex).
        It is used for better visualisation of square matrices
        '''
        n = len(square_matrix)
        for i in range(n):
                for j in range(n):
                        if np.abs(square_matrix[i][j]) < 1e-10:
                                square_matrix[i][j] = 0.
        return square_matrix

#%%###########################End Functions#############################

#%%########################QI OBSERVABLES###############################
def Get_Purity(rho: list[list[complex]]) -> float:
        '''
        Takes as input a square matrix.
        Calculate the trace of the square of the density matrix. The quantum state is pure if Tr[rho^2] = 1.
        '''
        rho2 = np.dot(rho, rho)
        return np.trace(rho2)

def Get_Bell_Test(CTC: list[list[float]]) -> tuple[float, bool]: #a tester
        '''
        This functions takes as an entry the matrix transpose(C) * C and returns the Bell inequality and the optimal directions.
        This is only working for pair of qubits for now
        '''

        eigvals, eigvecs = la.eigh(np.array(CTC)) #We now need to sort them
        sorted_eigs = eigvals.sort(reverse = True)
        crit = 1 - sorted_eigs[-1] - sorted_eigs[-2]
        return crit, crit < 0

def Get_Concurrence(rho: list[list[complex]]) -> float: #a tester
        """
        Peut etre amelioree, notamment en utilisant que la matrice est hermitienne pour eviter d'inverser des matrices de changement de base
        """
        if len(rho) != 4:
                raise ValueError('The length of the density matrix is not correct. This function only handles systems of 2 qubits!')
        
        aux = np.kron(sigma2, sigma2)
        rho_tilde = np.dot(aux, np.dot(np.conjugate(rho), aux)) # rhotilde = aux * rho* * aux
        #print('rho tilde', rho_tilde)
        eigvals, eigvecs = la.eigh(np.array(rho)) # diagonalization formula: M = P D P^{-1}
        for o in range(len(eigvals)): #This is necessary because numerical errors can cause a almost zero eigenvalue to be negative.
                if eigvals[o]/max(eigvals) < 1e-20:
                        eigvals[o] = 0.       

        sqrt_rho = np.dot(eigvecs, np.dot(np.sqrt(np.diag(eigvals)), la.inv(eigvecs)))

        rho_aux = np.dot(sqrt_rho, np.dot(rho_tilde, sqrt_rho))

        eigvals2, _ = la.eigh(rho_aux)
        for o in range(len(eigvals2)): #This is necessary because numerical errors can cause a almost zero eigenvalue to be negative.
                if eigvals2[o]/max(eigvals2) < 1e-20:
                        eigvals2[o] = 0.

        final_eigvals = np.sqrt(eigvals2) #the eigenvalues are > 0 because hermitian (are we sure that it is still hermitian after the transformation ?)
        final_eigs_sorted = np.sort(final_eigvals)

        Concurrence = max(0, final_eigs_sorted[3] - final_eigs_sorted[2] - final_eigs_sorted[1] - final_eigs_sorted[0]) # = max(0, lambda1 - lambda2 - lambda3 - lambda4)

        return Concurrence

def Get_Dcoef(C:list[list[float]]) -> tuple[float, float, float, float, bool]: #This functions depends a lot on the basis we use, put in comment the convention we use.
        """
        In MadGraph we use the momenta referential as: {n, r, k} with:
        k : top direction, r = (p - k cos(theta))/sin(theta), n = p x k/sin(theta)
        So C_nn = C[0][0], C_rr = C[1][1], C_kk = C[2][2]
        -> check with the rotation and boost functions if it is correct
        """
        if len(C) != 3:
                raise ValueError('The length of the correlation matrix is not correct. This function only handles systems of 2 qubits!')

        Done = np.real((C[0][0] + C[2][2] + C[2][2])/3.) #D1 = (C_nn + C_rr + C_kk)/3
        Dn = np.real((C[0][0] - C[1][1] - C[2][2])/3.) #Dn = -(-C_nn + C_rr + C_kk)/3
        Dr = np.real((-C[0][0] + C[1][1] - C[2][2])/3.) #Dr = -(C_nn - C_rr + C_kk)/3
        Dk = np.real((-C[0][0] - C[1][1] + C[2][2])/3.) #Dk = -(C_nn + C_rr - C_kk)/3
        boolD = Done < - 1/3 or Dn < - 1/3 or Dr < - 1/3 or Dk < - 1/3 #if boolD = True, then there is entanglement
        return Done, Dn, Dr, Dk, boolD

def Get_Concurrence_C(C: list[list[float]]) -> float:
        """
        This function is faster if we already have calculated the correlation matrix.
        It only works if the spin-correlation matrix is diagonal
        """
        if len(C) != 3:
                raise ValueError('The length of the correlation matrix is not correct. This function only handles systems of 2 qubits!')

        Done, Dn, Dr, Dk, _ = Get_Dcoef(C)
        Dmin = min(Done, Dn, Dr, Dk)

        return max(0, -1 - 3*np.real(Dmin))/2

# def Shannon_Entropy(p:float) -> float:
#         return -p * np.log2(p) - (1 - p) * np.log2(1-p)

# def Get_Ent_Form(Concurrence:float) -> float:
#         E = Shannon_Entropy((1 + np.sqrt(1 - Concurrence**2))/2)
#         return E
#########################END QI OBSERVABLES ############################
def permutations_PGD(PDG:list[int], status:list[int])-> list[list[int]]:
        """
        Input: a list of PDGs + a list of status
        Output: all the possible PDGs permutations keeping incoming and outcoming particles separate
        """
        nincoming, noutcoming = 0, 0
        End = []

        for i in range(len(status)):
                if status[i] == -1:
                        nincoming += 1
                elif status[i] == +1:
                        noutcoming +1
                else:
                        raise ValueError("Status not correct")

        InitialState = PDG[0:nincoming]
        FinalState = PDG[nincoming:]
        All_InitialState = list(set(list(permutations(InitialState))))
        All_FinalState = list(set(list(permutations(FinalState))))
        
        list_initial_states = [list(All_InitialState[i]) for i in range(len(All_InitialState))]
        list_final_states = [list(All_FinalState[i]) for i in range(len(All_FinalState))]
        

        for i in range(len(list_initial_states)):
                for j in range(len(list_final_states)):
                        End.append(list_initial_states[i] + list_final_states[j])

        return End