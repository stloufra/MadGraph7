import numpy as np
from numpy import linalg as la
from itertools import permutations
from typing import Union

#%% Global variables for fermions and bosons
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


#%%#######################Auxiliary functions###########################
def square_matrix(line_matrix: list[complex]) -> list[complex]:
        """
        Input: hermitian matrix in a line form
        Output: hermitian matrix in the traditional square form
        This function converts a hermitian matrix expressed as a line into a typical matrix form.
        Caution : the definition of matrices in fortran and python are not the same.
        """
        length_lin = len(line_matrix)
        n = (-1 + np.sqrt(1 + 8*length_lin))/2
        if n%1 != 0:
                raise ValueError("Error :The line matrix given has incorrect length")
        else:
                n = int(n)

        matrix_square = np.zeros((n, n), dtype=np.complex128)

        for i in range(n):
                for k in range(n):
                        if k > i:
                                matrix_square[i][k] = np.conjugate(line_matrix[i*n + k - i*(i + 1)//2]) 
                        elif k == i:
                                matrix_square[i][k] = line_matrix[i*n + k - i*(i + 1)//2] #this line is just here because the diagonal elements should not be conjugated (they are real anyway)
                        else:
                                matrix_square[i][k] = np.conjugate(matrix_square[k][i])
                                                                
        return matrix_square


def write_density_matrix(rho: list[complex], pos: list[int], allow_hel: list[int], epsilon=1e-10) -> None:
        """
        Input:  rho -> the density matrix in line format
                pos -> position of the particles that you put in the density matrix
                allow_hel -> helicity basis chosen for the density matrix
                epsilon -> threshold parameter
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

def get_trace(line_matrix: list[complex], n_comb: int) -> float:
        """
        Input:  line_matrix -> density matrix in its line format
                n_comb -> number of helicity combinations possible for the particles in the density matrix
        Output: the trace of the density matrix in its line format
        """
        diagIndices = [0] * n_comb
        counter = 0
        for idiag in range(n_comb):
               diagIndices[idiag] = idiag * n_comb - counter
               counter += idiag
        trace = 0
        for i in range(n_comb):
               trace += np.real(line_matrix[diagIndices[i]])

        return trace

def get_rho_normalised(rho: list[complex], n_comb: int, epsilon=1e-10) -> list[complex]:
        """
        Input:  rho -> density matrix in its line format
                n_comb -> number of helicity combinations possible for the particles in the density matrix
                epsilon -> 
        Output: the trace of the density matrix in its line format
        """
        trace = get_trace(rho, n_comb)

        for i in range(len(rho)):
                if abs(rho[i].real / trace) < epsilon:
                        rho[i] = 0. + rho[i].imag *1j
                if abs(rho[i].imag / trace) < epsilon:
                        rho[i] = rho[i].real + 0. *1j

        aux = np.array(rho) / trace
        return aux

def get_list_sliced(rho_list: list[complex], n_comb: int, epsilon = 1e-10) -> list[complex]:
        """
        Input:  rho -> density matrix in its line format
                n_comb -> number of helicity combinations possible for the particles in the density matrix
                epsilon -> threshold parameter
        Output: rho_sliced -> the density matrix in line format with the correct length

        The fortran code defines list of length 99. This function removes 
        the elements that are not relevant for a more practical use. 
        """
        if isinstance(n_comb, int) :
                rho_list = rho_list[0: int(n_comb*(n_comb + 1)/2)]
                trace = get_trace(rho_list, n_comb)
                for i in range(len(rho_list)):
                        if abs(rho_list[i].real / trace) < epsilon:
                                rho_list[i] = 0. + rho_list[i].imag *1j
                        if abs(rho_list[i].imag / trace) < epsilon:
                                rho_list[i] = rho_list[i].real + 0. *1j
                return rho_list
        else:
                return "Error: n_comb is not an integer"


def Get_Correlations(rho:list[complex, complex], pdg_pos:list[int], epsilon=1e-10) -> tuple[list[float, float], list[float, float]]:
        """
        Input:  rho -> density matrix in its matrix format
                pdg_pos -> list of pdg codes of the particles in the density matrix
                epsilon -> threshold parameter
        Output: Correlations matrix C and the square C^T . C

        This function takes as input a density matrix in a square (python) format. 
        It then calculates the correlation matrix of the spin of the two 
        fermions/two vector bosons with the trace formula.
        """
        fermions_pdg = [1, 2, 3, 4, 5, 6, 7, 8] + [-1, -2, -3, -4, -5, -6, -7, -8] + [11, 12, 13, 14, 15, 16, 17, 18] + [-11, -12, -13, -14, -15, -16, -17, -18]
        boson_pdg = [9, 21, 22, 23, 24, -24] #these 3 entries could be put as inputs to generalise to any model, for instance with Higgs characterozation one
        scalar_pdg = [25, 37]
        if pdg_pos[0] in fermions_pdg: 
                if pdg_pos[1] in fermions_pdg: #both particles are fermion
                        Correlations = np.zeros((3, 3), dtype=float)
                        for i in range(3):
                                for j in range(3):
                                        tensorprod = np.kron(sigma[i], sigma[j])
                                        Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, rho)))
                                        if np.abs(Correlations[i][j]) < epsilon:
                                                Correlations[i][j] = 0.
                elif pdg_pos[1] in boson_pdg: #1st particle fermion, 2nd massive boson
                        Correlations = np.zeros((3, 8), dtype=float)
                        for i in range(3):
                                for j in range(8):
                                        tensorprod = np.kron(sigma[i], Lambda[j])
                                        Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, rho))/2.)
                                        if np.abs(Correlations[i][j]) < epsilon:
                                                Correlations[i][j] = 0.
                elif pdg_pos[1] in scalar_pdg: #1st particle fermion, 2nd scalar
                        raise ValueError('There is no spin-correlation with a scalar')
                else:
                        raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')

        elif pdg_pos[0] in boson_pdg:
                if pdg_pos[1] in fermions_pdg: #1st particle massive boson, 2nd fermion
                       Correlations = np.zeros((8, 3), dtype=float)
                       for i in range(8):
                                for j in range(3):
                                        tensorprod = np.kron(Lambda[i], sigma[j])
                                        Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, rho))/2.)
                                        if np.abs(Correlations[i][j]) < epsilon:
                                                Correlations[i][j] = 0.
                elif pdg_pos[1] in boson_pdg: #both particles are massive boson
                        Correlations = np.zeros((8, 8), dtype=float)
                        for i in range(8):
                                for j in range(8):
                                        tensorprod = np.kron(Lambda[i], Lambda[j]) #the tensor product is with Gell-Mann matrices
                                        Correlations[i][j] = np.real(np.trace(np.dot(tensorprod, rho))/4.) #there is an additional 1/4 factor
                                        if np.abs(Correlations[i][j]) < epsilon:
                                                Correlations[i][j] = 0.
                elif pdg_pos[1] in scalar_pdg: #1st particle massive boson, 2nd scalar
                        raise ValueError('There is no spin-correlation with a scalar')
                else:
                        raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')              
        elif pdg_pos[0] in scalar_pdg:
               raise ValueError('There is no spin-correlation with a scalar')
        else:
                raise ValueError('One of the particle is not a scalar/fermion/vector boson, the code cant handle it yet')     

        Correlations_squared = np.dot(np.transpose(Correlations),Correlations)

        return Correlations, Correlations_squared

#%%This functon computes the polarisation of the qubits for square density matrix
def Get_Polarisations(rho:list[complex, complex], pdg_pos:list[int], epsilon=1e-10) -> tuple[list[float], list[float]]:
        """
        Input:  rho -> density matrix in its matrix format
                pdg_pos -> list of pdg codes of the particles in the density matrix
                epsilon -> threshold parameter
        Output: Polarisations of both particles

        This function takes as input a density matrix in a square (python) format. 
        It then calculates the polarisations of the two fermions/two vector bosons
        using the trace formula.
        """
        fermions_pdg = [1, 2, 3, 4, 5, 6, 7, 8] + [-1, -2, -3, -4, -5, -6, -7, -8] + [11, 12, 13, 14, 15, 16, 17, 18] + [-11, -12, -13, -14, -15, -16, -17, -18]
        boson_pdg = [9, 21, 22, 23, 24, -24] 
        scalar_pdg = [25, 37]
        #pdg_pos[0] is the first particle
        if pdg_pos[0] in fermions_pdg: #the first particle is a fermion or a massless vector boson -> it has 2 spin states possibles
                Polarisation1 = np.zeros(3, dtype=np.complex128)
                if pdg_pos[1] in fermions_pdg: #1st particle fermion, 2nd fermion
                        Polarisation2 = np.zeros(3, dtype=np.complex128)
                        for i in range(3):
                                tensorprod1 = np.kron(sigma[i], Identity2)
                                tensorprod2 = np.kron(Identity2, sigma[i])
                                Polarisation1[i] = np.trace(np.dot(tensorprod1, rho))
                                Polarisation2[i] = np.trace(np.dot(tensorprod2, rho))
                elif pdg_pos[1] in boson_pdg: #1st particle fermion, 2nd massive vector boson
                        Polarisation2 = np.zeros(8, dtype=np.complex128)
                        for i in range(3):
                                tensorprod1 = np.kron(sigma[i], Identity3)
                                Polarisation1[i] = np.trace(np.dot(tensorprod1, rho))
                        for j in range(8):
                                tensorprod2 = np.kron(Identity2, Lambda[j])
                                Polarisation2[j] = np.trace(np.dot(tensorprod2, rho))/2.
                else:
                        raise ValueError("One of the particle chosen is either a scalar or not recognised")

        elif pdg_pos[0] in boson_pdg: #massive vector boson -> it has 3 spin states possibles
                Polarisation1 = np.zeros(8, dtype=np.complex128)
                if pdg_pos[1] in fermions_pdg: #1st particle massive vector boson, 2nd fermion
                        Polarisation2 = np.zeros(3, dtype=np.complex128)
                        for i in range(8):
                                tensorprod1 = np.kron(Lambda[i], Identity2)
                                Polarisation1[i] = np.trace(np.dot(tensorprod1, rho))/2.
                        for j in range(3):
                                tensorprod2 = np.kron(Identity3, sigma[j])
                                Polarisation2[j] = np.trace(np.dot(tensorprod2, rho))
                elif pdg_pos[1] in boson_pdg: #1st particle massive vector boson, 2nd massive vector boson
                        Polarisation2 = np.zeros(8, dtype=np.complex128)
                        for i in range(8):
                                tensorprod1 = np.kron(Lambda[i], Identity3)
                                tensorprod2 = np.kron(Identity3, Lambda[i])
                                Polarisation1[i] = np.trace(np.dot(tensorprod1, rho))/2.
                                Polarisation2[i] = np.trace(np.dot(tensorprod2, rho))/2.
                else:
                        raise ValueError("One of the particle chosen is either a scalar or not recognised")

        else:
                raise ValueError('One of the particle is not a fermion/vector boson, the code cant handle it yet')


        if np.abs(Polarisation1[i].real) < epsilon:
                Polarisation1[i] = 0. + Polarisation1[i].imag*1j
        if np.abs(Polarisation1[i].imag) < epsilon:
                Polarisation1[i] = Polarisation1[i].real + 0.*1j
        
        if np.abs(Polarisation2[i].real) < epsilon:
                Polarisation2[i] = 0. + Polarisation2[i].imag*1j
        if np.abs(Polarisation2[i].imag) < epsilon:
                Polarisation2[i] = Polarisation2[i].real +0.*1j

        return Polarisation1, Polarisation2


def spin_expectation(Polarisation1: list[float], Polarisation2: list[float]) -> tuple[list[float], list[float]]:
        """
        Input: Polarisation1, Polarisation2 -> polarisation vectors for the two particles studied
        Output: exp1, exp2 -> the spin expectation values for each particle
        """
        spin_exp1 = np.zeros(3, dtype=np.complex128)
        spin_exp2 = np.zeros(3, dtype=np.complex128)

        if len(Polarisation1) == 3:
            if len(Polarisation2) == 3: #both particles are fermions
                spin_exp1 = np.array(Polarisation1)/2.
                spin_exp2 = np.array(Polarisation2)/2.
            elif len(Polarisation2) == 8: #first particle fermion second particle boson
                spin_exp1 = np.array(Polarisation1)
                spin_exp2[0] = np.sqrt(2) * (Polarisation2[0] + Polarisation2[5])
                spin_exp2[1] = np.sqrt(2) * (Polarisation2[1] + Polarisation2[6])
                spin_exp2[2] = Polarisation2[2] + np.sqrt(3) * Polarisation2[7]
        elif len(Polarisation1) == 8: 
            if len(Polarisation2) == 3: #first particle boson, second particle fermion
                spin_exp1[0] = np.sqrt(2) * (Polarisation1[0] + Polarisation1[5])
                spin_exp1[1] = np.sqrt(2) * (Polarisation1[1] + Polarisation1[6])
                spin_exp1[2] = Polarisation1[2] + np.sqrt(3) * Polarisation1[7]
                spin_exp2 = np.array(Polarisation2)
            elif len(Polarisation2) == 8: #both particles are bosons
                spin_exp1[0] = np.sqrt(2) * (Polarisation1[0] + Polarisation1[5])
                spin_exp1[1] = np.sqrt(2) * (Polarisation1[1] + Polarisation1[6])
                spin_exp1[2] = Polarisation1[2] + np.sqrt(3) * Polarisation1[7]
                spin_exp2[0] = np.sqrt(2) * (Polarisation2[0] + Polarisation2[5])
                spin_exp2[1] = np.sqrt(2) * (Polarisation2[1] + Polarisation2[6])
                spin_exp2[2] = Polarisation2[2] + np.sqrt(3) * Polarisation2[7]
                
             
        else:
              raise ValueError("The code only deals with fermions and bosons for now. Please change the polarisations inputed.")
        
        return np.real(spin_exp1), np.real(spin_exp2)

def spinspin_expectation(Correlation: list[float]) -> list[float, float]:
        """
        Input: Polarisation1, Polarisation2 -> polarisation vectors for the two particles studied
        Output: exp1, exp2 -> the spin expectation values for each particle
        """
        spin_spin_exp = np.zeros((3,3), dtype=np.complex128)
        Corr = np.array(Correlation)

        if Corr.shape == (3,3): #2 fermions
              spin_spin_exp = Corr/4.

        elif Corr.shape == (8,8): #2 bosons
              spin_spin_exp[0][0] = 2 * (Corr[0][0] + Corr[0][5] + Corr[5][0] + Corr[5][5])
              spin_spin_exp[0][1] = 2 * (Corr[0][1] + Corr[0][6] + Corr[5][1] + Corr[5][6])
              spin_spin_exp[0][2] = np.sqrt(2) *(Corr[0][2] + Corr[5][2] + np.sqrt(3) * (Corr[0][7] + Corr[5][7]))

              spin_spin_exp[1][0] = 2 * (Corr[1][0] + Corr[6][0] + Corr[1][5] + Corr[6][5])
              spin_spin_exp[1][1] = 2 * (Corr[1][1] + Corr[1][6] + Corr[6][1] + Corr[6][6])
              spin_spin_exp[1][2] = np.sqrt(2) *(Corr[1][2] + Corr[6][2] + np.sqrt(3) * (Corr[1][7] + Corr[6][7]))

              spin_spin_exp[2][0] = np.sqrt(2) * (Corr[2][0] + Corr[2][5] + np.sqrt(3) * (Corr[7][0] + Corr[7][5]))
              spin_spin_exp[2][1] = np.sqrt(2) * (Corr[2][1] + Corr[2][6] + np.sqrt(3) * (Corr[7][1] + Corr[7][6]))
              spin_spin_exp[2][2] = Corr[2][2] + np.sqrt(3) * (Corr[2][7] + Corr[7][2]) + 3 * Corr[7][7]
        
        elif Corr.shape == (3,8): #first particle fermion, second particle boson
              spin_spin_exp[0][0] = (Corr[0][0] + Corr[0][5]) / np.sqrt(2)
              spin_spin_exp[0][1] = (Corr[0][1] + Corr[0][6]) / np.sqrt(2)
              spin_spin_exp[0][2] = (Corr[0][2] + np.sqrt(3) * Corr[0][7]) / 2.

              spin_spin_exp[1][0] = (Corr[1][0] + Corr[1][5]) / np.sqrt(2)
              spin_spin_exp[1][1] = (Corr[1][1] + Corr[1][6]) / np.sqrt(2)
              spin_spin_exp[1][2] = (Corr[1][2] + np.sqrt(3) * Corr[1][7]) / 2.

              spin_spin_exp[2][0] = (Corr[2][0] + Corr[2][5]) / np.sqrt(2)
              spin_spin_exp[2][1] = (Corr[2][1] + Corr[2][6]) / np.sqrt(2)
              spin_spin_exp[2][2] = (Corr[2][2] + np.sqrt(3) * Corr[2][7]) / 2.

        elif Corr.shape == (8,3): #first particle boson, second particle fermion
              spin_spin_exp[0][0] = (Corr[0][0] + Corr[5][0]) / np.sqrt(2)
              spin_spin_exp[0][1] = (Corr[0][1] + Corr[5][1]) / np.sqrt(2)
              spin_spin_exp[0][2] = (Corr[0][2] + Corr[5][2]) / np.sqrt(2)

              spin_spin_exp[1][0] = (Corr[1][0] + Corr[6][0]) / np.sqrt(2)
              spin_spin_exp[1][1] = (Corr[1][1] + Corr[6][1]) / np.sqrt(2)
              spin_spin_exp[1][2] = (Corr[1][2] +  Corr[6][2]) / np.sqrt(2)

              spin_spin_exp[2][0] = (Corr[2][0] + np.sqrt(3) * Corr[7][0]) / 2.
              spin_spin_exp[2][1] = (Corr[2][1] + np.sqrt(3) * Corr[7][1]) / 2.
              spin_spin_exp[2][2] = (Corr[2][2] + np.sqrt(3) * Corr[7][2]) / 2.
        else:
              raise ValueError("The code only deals with fermions and bosons for now. Please change the correlation matrix inputed.")

        return np.real(spin_spin_exp)

def Partial_Trace(rho:list[complex, complex], index:int, pdg_pos:list[int]) -> list[complex, complex]:
    """
    Input: rho -> density matrix in matrix format
           index -> number of the subspace to trace over (1 or 2)
           pdg_pos -> list of pdg codes of the particles in the density matrix 
    Output: TracedRho -> the density matrix with one of its subspace traced over

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
        if abs(pdg_pos[0]) <= 22 and (abs(pdg_pos[1]) == 23 or abs(pdg_pos[1]) == 24): #fermion 1st particle, boson second particle
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
                
        elif (abs(pdg_pos[0]) == 23 or abs(pdg_pos[0]) == 24) and abs(pdg_pos[1]) <= 22: #boson 1st particle, fermion second particle
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



#%% Rotation for p = p[E, px, py, pz]
#it must be applied on momenta in the python format (so before invert_momenta)
def rotation(p: list[float], phi: float, theta: float, epsilon: float) ->list[float]:
        """
        Input:  p -> 4-momentum of a particle
                phi, theta -> angles for the rotation
                epsilon -> threshold parameter
        Output: pRot -> the 4-momentum of the particle rotated
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
        Inputs: p(0:3) -> four-momentum p in the q rest  frame
                q(0:3) -> four-momentum q in the boosted frame
                m -> mass of q (for numerical stability)
        Outputs:
                pboost -> 4-momentum of the particle boosted
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

def cut_matrix(square_matrix: list[complex, complex]) -> list[complex, complex]:
        '''
        Input:  matrix in python format
        Output: matrix in python format after the threshold cut
        Removes from a square matrix the terms that are smaller than 1e-10 (or their norm if complex).
        It is used for better visualisation of square matrices
        '''
        n = len(square_matrix)
        for i in range(n):
                for j in range(n):
                        if np.abs(square_matrix[i][j]) < 1e-10:
                                square_matrix[i][j] = 0.
        return square_matrix

def opp_momentum(p: list[float]) -> list[float]:
    return [p[0], -p[1], -p[2], -p[3]]

def norm_momentum(p: list[float]) -> float:
    return np.sqrt(p[0]**2 - p[1]**2 - p[2]**2 - p[3]**2)

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

def plot_hist(x:list[float], y:list[float], z:list[float], limitx:list[float], limity:list[float], n_binx:int, n_biny:int) -> tuple[list[float, float], list[float, float]]:
    """
    Input: x, y, z -> data
           limitx, limity -> bounds for the x and y-axis
           n_binx, n_biny -> number of bins for the x and y-axis
    Output: matrix of the 2D histogram, matrix of the number of events

    x, y, z must be vectors of same length
    The structure of the two matrices are : origin lower left, growing right and up
    """
    binsx = np.linspace(limitx[0], limitx[1], n_binx + 1)
    binsy = np.linspace(limity[0], limity[1], n_biny + 1)
    Map = np.zeros((n_biny, n_binx))
    N_Map = np.zeros((n_biny, n_binx))
    for k in range(len(z)):
            for i in range(len(binsx) - 1): #we need the -1 because we added +1 when defining binsx
                    if x[k] >= binsx[i] and x[k] < binsx[i + 1]: #this means that the second index of Map is i
                            for j in range(len(binsy) - 1):
                                    if y[k] >= binsy[j] and y[k] < binsy[j + 1]: #this means that the first index of Map is j
                                            Map[j][i] += z[k] #if x and y are in the correct bin, we add the value of z in the bin.
                                            N_Map[j][i] += 1

    Number_Events = len(z)
    for i in range(len(Map)):
          for j in range(len(Map[0])):
                if N_Map[i][j] == 0:
                      Map[i][j] = np.nan
                else:
                      Map[i][j] = Map[i][j]/ N_Map[i][j]
                      N_Map[i][j] = N_Map[i][j]/Number_Events

    return Map, N_Map

def permutations_PGD(PDG: list[int], status: list[int])-> list[list[int]]:
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


def add_phase_density(rho: list[complex, complex], phases: list[float]) ->list[complex, complex]:
    """
        Input:  rho -> density matrix in matrix format
                phases -> phases to add for each helicity eigenvector
        Output: density matrix with the phases added
        This function adds global phases on each component of the spinor/polarisation vector
       It is used to compare to analytical results. To compare to default Feyncalc conventions,
       one can set phases = [0, np.pi, 0, 0]
    """
    alpha = phases[0]
    beta = phases[1]
    gamma = phases[2]
    delta = phases[3]
    U = np.diag([np.exp((alpha + delta)*1j), np.exp((alpha + gamma)*1j), np.exp((beta + delta)*1j), np.exp((beta + gamma)*1j)])
    aux = np.dot(rho, U)
    rho_corrected = np.dot(np.conjugate(U), aux)
    for i in range(len(rho_corrected)):
        for j in range(len(rho_corrected)):
            if abs(rho_corrected[i][j].real) < 1e-10:
                rho_corrected[i][j] = 0. + rho_corrected[i][j].imag * 1j
            if abs(rho_corrected[i][j].imag) < 1e-10:
                rho_corrected[i][j] = rho_corrected[i][j].real + 0. * 1j
    
    return rho_corrected

def add_phase_density_qutrits(rho: list[complex, complex], phases: list[complex]) ->list[complex, complex]:
    """
        Input:  rho -> density matrix in matrix format
                phases -> phases to add for each helicity eigenvector
        Output: density matrix with the phases added
        This function adds global phases on each component of the spinor/polarisation vector
        It is used to compare to analytical results. To compare to default Feyncalc conventions,
        one can set phases = [0, -np.pi/2, 0, 0, -np.pi/2, 0]
    """
    alpha = phases[0]
    beta = phases[1]
    gamma = phases[2]
    delta = phases[3]
    epsilon = phases[4]
    tau = phases[5]
    U = np.diag([np.exp((alpha + tau)*1j), np.exp((alpha + epsilon)*1j), np.exp((alpha + delta)*1j), np.exp((beta + tau)*1j), np.exp((beta + epsilon)*1j), np.exp((beta + delta)*1j), np.exp((gamma + tau)*1j), np.exp((gamma + epsilon)*1j), np.exp((gamma + delta)*1j)])
    
    aux = np.dot(rho, U)
    rho_corrected = np.dot(np.conjugate(U), aux)
    
    for i in range(len(rho_corrected)):
        for j in range(len(rho_corrected)):
            if abs(rho_corrected[i][j].real) < 1e-10:
                rho_corrected[i][j] = 0. + rho_corrected[i][j].imag * 1j
            if abs(rho_corrected[i][j].imag) < 1e-10:
                rho_corrected[i][j] = rho_corrected[i][j].real + 0. * 1j
    
    return rho_corrected

#%%###########################End auxiliary functions###################

#%%########################QI OBSERVABLES###############################
def Get_Purity(rho: list[complex, complex]) -> float:
        '''
        Input: rho -> density matrix in matrix format
        Output: purity
        The quantum state is pure if Tr[rho^2] = 1.
        '''
        rho2 = np.dot(rho, rho)
        Purity = np.trace(rho2)
        return Purity.real

def ConcLB2(Rho: list[complex, complex], pdg_pos= list[int])-> float:
    """
    Input:  rho -> density matrix in matrix format
            pdg_pos -> pdg codes of the particles in the density matrix
    Output: square of the lower bound of the concurrence for a system composed of a pair of qutrits.
    """

    RhoA = Partial_Trace(Rho, 2, pdg_pos)
    RhoB = Partial_Trace(Rho, 1, pdg_pos)

    if ((np.trace(RhoA) - 1) > 1e-10) or ((np.trace(RhoB) - 1) > 1e-10):
        print('Warning: the traced-out density matrices have non unitary trace!')

    aux1 = np.trace(np.dot(Rho, Rho)) - np.trace(np.dot(RhoA, RhoA))
    aux2 = np.trace(np.dot(Rho, Rho)) - np.trace(np.dot(RhoB, RhoB))
    ConcLB2 = 2 * max(0, aux1, aux2)

    return ConcLB2.real

def ConcUB2(Rho: list[complex, complex], pdg_pos: list[int])-> float:
    """
    Input:  rho -> density matrix in matrix format
            pdg_pos -> pdg codes of the particles in the density matrix
    Output: square of the upper bound of the concurrence for a system composed of a pair of qutrits.
    """

    RhoA = Partial_Trace(Rho, 2, pdg_pos)
    RhoB = Partial_Trace(Rho, 1, pdg_pos)

    if ((np.trace(RhoA) - 1) > 1e-10) or ((np.trace(RhoB) - 1) > 1e-10):
        print('Warning: the traced-out density matrices have non unitary trace!')

    aux1 = 1 - np.trace(np.dot(RhoA, RhoA))
    aux2 = 1 - np.trace(np.dot(RhoB, RhoB))
    ConcUB2 = 2 * min(aux1, aux2)
    return ConcUB2.real

def Get_Bell_Test(CTC: list[list[float]]) -> tuple[float, bool]: #a tester
        '''
        This functions takes as an entry the matrix transpose(C) * C and returns the Bell inequality and the optimal directions.
        This is only working for pair of qubits.
        '''

        eigvals, eigvecs = la.eigh(np.array(CTC)) #We now need to sort them
        sorted_eigs = eigvals.sort(reverse = True)
        crit = 1 - sorted_eigs[-1] - sorted_eigs[-2]
        return crit, crit < 0

def Get_Concurrence(rho: list[complex, complex]) -> float:
        """
        Input:  rho -> density matrix of a system qubit/qubit
        Output: concurrence
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
        Input:  C -> spin correlations matrix
        Output: D1, Dr, Dn, Dk -> the 4 D-coefficients
                boolD -> True if there entanglement
        In MadGraph we use the momenta referential as: {n, r, k} with:
        k : top direction, r = (p - k cos(theta))/sin(theta), n = p x k/sin(theta)
        So C_nn = C[0][0], C_rr = C[1][1], C_kk = C[2][2]
        """
        if len(C) != 3:
                raise ValueError('The length of the correlation matrix is not correct. This function only handles systems of 2 qubits!')

        Done = np.real((C[0][0] + C[1][1] + C[2][2])/3.) #D1 = (C_nn + C_rr + C_kk)/3
        Dn = np.real((C[0][0] - C[1][1] - C[2][2])/3.) #Dn = -(-C_nn + C_rr + C_kk)/3
        Dr = np.real((-C[0][0] + C[1][1] - C[2][2])/3.) #Dr = -(C_nn - C_rr + C_kk)/3
        Dk = np.real((-C[0][0] - C[1][1] + C[2][2])/3.) #Dk = -(C_nn + C_rr - C_kk)/3
        boolD = Done < - 1/3 or Dn < - 1/3 or Dr < - 1/3 or Dk < - 1/3 #if boolD = True, then there is entanglement
        return Done, Dn, Dr, Dk, boolD

def Get_Concurrence_C(C: list[list[float]]) -> float:
        """
        Input:  C -> spin correlations matrix
        Output: Concurrence
        This function is faster if we already have calculated the correlation matrix.
        Note: it only works if the spin-correlation matrix is diagonal
        """
        if len(C) != 3:
                raise ValueError('The length of the correlation matrix is not correct. This function only handles systems of 2 qubits!')

        Done, Dn, Dr, Dk, _ = Get_Dcoef(C)
        Dmin = min(Done, Dn, Dr, Dk)

        return max(0, -1 - 3*np.real(Dmin))/2

def Shannon_Entropy(p:float) -> float:
        return -p * np.log2(p) - (1 - p) * np.log2(1-p)

def Get_Ent_Form(Concurrence:float) -> float:
        E = Shannon_Entropy((1 + np.sqrt(1 - Concurrence**2))/2)
        return E

def get_Pauli_string(n:int) ->list[list[complex, complex]]:
        """
        Input:  n -> dimension of the Pauli string
        Output  List of the elements of the Pauli string
        Takes an integer n and returns all the Pauli strings for a system of n fermions
        """
        P1 = [Identity2]
        for i in range(len(sigma)):
                P1.append(sigma[i])
        
        Pauli_string = []

        if n == 1:
                Pauli_string = P1
                return Pauli_string
        else:
                for elem1 in P1:
                        for elem2 in get_Pauli_string(n-1):
                                Pauli_string.append(np.kron(elem1, elem2))
                return Pauli_string


def Magic_Pure(rho: list[complex, complex], n: int) -> complex:
        """
        Input:  rho -> density matrix in matrix format
                n -> number of qubits (usually taken as 2)  
        Output: magic
        Computes the quantity M2 for a density matrix which represents a pure state for a system of n qubits
        """
        Xi = 0
        Pauli_strings = get_Pauli_string(n)
        for P in Pauli_strings:
                Xi += np.trace(np.dot(P, rho)) ** 4
        return - np.log2(Xi / 2**n)

def Magic_Mixed(rho: list[complex, complex], n: int) -> complex:
        """
        Input:  rho -> density matrix in matrix format
                n -> number of qubits (usually taken as 2)  
        Output: magic
        Computes the quantity M2 for a density matrix which represents a mixed state for a system of n qubits
        """
        XiNum = 0
        XiDenom = 0
        Pauli_strings = get_Pauli_string(n)
        for P in Pauli_strings:
                XiNum += np.trace(np.dot(P, rho)) ** 4
                XiDenom += np.trace(np.dot(P, rho)) ** 2
        Magic = - np.log2(XiNum / XiDenom)
        return Magic.real

def Partial_Transpose(rho:list[complex, complex], index:int, pdg_code:list[int]) ->list[complex, complex]:
    """
    Input:  rho -> density matrix
            index -> integer designating the subspace to transpose (1 or 2)
            pdg_code -> list of 2 elements, the pdg code of both particles in the density matrix
    Output: the density matrix partially transposed
    """
    fermions_pdg = [1, 2, 3, 4, 5, 6, 7, 8] + [-1, -2, -3, -4, -5, -6, -7, -8] + [11, 12, 13, 14, 15, 16, 17, 18] + [-11, -12, -13, -14, -15, -16, -17, -18]
    boson_pdg = [9, 21, 22, 23, 24, -24] #these 3 entries could be put as inputs to generalise to any model, for instance with Higgs characterozation one
    scalar_pdg = [25, 37]

    if len(pdg_code) != 2:
        raise ValueError("The pdg_code input must be of length 2.")

    #Basis1 is the basis of the dimension of the first particle, Basis2 is the same for the second particle
    if pdg_code[0] in fermions_pdg:
        n1 = 2
    elif pdg_code[0] in boson_pdg:
        n1 = 3
    else:
        raise ValueError("The pdg of the first particle is not correct (either designs a scalar particle or a non recognised particle). Please change it.")

    if pdg_code[1] in fermions_pdg:
        n2 = 2
    elif pdg_code[1] in boson_pdg:
        n2 = 3
    else:
        raise ValueError("The pdg of the second particle is not correct (either designs a scalar particle or a non recognised particle). Please change it.")

    Basis1 = np.identity(n1)
    Basis2 = np.identity(n2)

    # Reference : rho_line.append(np.dot(np.kron(Basis3[i], Basis3[j]), np.dot(rho, np.kron(Basis3[k], Basis3[l]))))
    rho_line = []

    if index == 0: #no change
        for i in range(len(Basis1)):
            for j in range(len(Basis2)):
                for k in range(len(Basis1)):
                    for l in range(len(Basis2)):
                        rho_line.append(np.dot(np.kron(Basis1[i], Basis2[j]), np.dot(rho, np.kron(Basis1[k], Basis2[l]))))  # exchanging j and l is transpose over A
                                                                                                                            # exchanging i and k is transpose over B
    elif index == 1: #transpose over first subspace
        for i in range(len(Basis1)):
            for j in range(len(Basis2)):
                for k in range(len(Basis1)):
                    for l in range(len(Basis2)):
                        rho_line.append(np.dot(np.kron(Basis1[i], Basis2[l]), np.dot(rho, np.kron(Basis1[k], Basis2[j]))))
    elif index == 2:
        for i in range(len(Basis1)):
            for j in range(len(Basis2)):
                for k in range(len(Basis1)):
                    for l in range(len(Basis2)):
                        rho_line.append(np.dot(np.kron(Basis1[k], Basis2[j]), np.dot(rho, np.kron(Basis1[i], Basis2[l]))))
    else:
        raise ValueError("The value of index is wrong, please use 1 to transpose over the first subspace or 2 to transpose over the second.")                   

    return np.array(rho_line).reshape(n1*n2,n1*n2)

def Negativity(rho:list[complex, complex], pdg_pos:list[int]) -> tuple[float, float]:
        """
           Input:  rho -> density matrix
                   pdg_pos -> list of the pdg code of the particles in the density matrix
           Output: negativity
           This functions computes the negativity of a density matrix composed of any two particles. From https://arxiv.org/abs/quant-ph/0504163
        """

        #We take the partial transpose of the second particle because the negativity does not depend on it 
        rho_TB = Partial_Transpose(rho, 2, pdg_pos)
        aux = np.dot(np.transpose(np.conjugate(rho_TB)), rho_TB)

        eigvals, eigvecs = la.eigh(aux) # diagonalization formula: M = P D P^{-1}
        sqrt_rho = np.dot(eigvecs, np.dot(np.sqrt(np.diag(eigvals)), la.inv(eigvecs)))

        Negativity = (np.trace(sqrt_rho) - 1)/2.
        Log_Negativity = np.log2(np.trace(sqrt_rho))
        return Negativity.real, Log_Negativity.real

def shift_clock(d: int) -> tuple[list[complex, complex], list[float, float]]:
    """
       Input:  d -> dimension of the Hilbert space of the chosen particle. Fermion = 2, Massive boson = 3.
       Output: the clock operator Z and the shift operator X.
    """
    RootsUnity = [np.exp(2*np.pi*k*1j/d) for k in range(d)]
    Z = np.diag(RootsUnity)
    for i in range(len(Z)):
        for j in range(len(Z)):
            if abs(Z[i][j].real) < 1e-10:
                Z[i][j] = 0. + Z[i][j].imag *1j
            if abs(Z[i][j].imag) < 1e-10:
                Z[i][j] = Z[i][j].real + 0. *1j
        
    X = np.diag([1 for i in range(d - 1)], -1)
    X[0][d - 1] = 1
    return Z, X

def Displacement_Operator(d: int) -> list[list[complex, complex]]:
    """
        Input:  d -> dimension of the Hilbert space of the chosen particle
        Output: list of all the displacement operator
        The convention in the ordering of the indices follows https://arxiv.org/abs/2003.02717
        Note: the convention differs from the one of qbism.
    """
    Displacement_list = []

    Z, X = shift_clock(d)
    RootUnity = np.exp(2*np.pi*1j/d)
    for u in range(d):
        for v in range(d):
            Prefactor = RootUnity ** (u * v * (d + 1) / 2)
            Xu = la.matrix_power(X, u)
            Zv = la.matrix_power(Z, v)
            D = Prefactor * np.dot(Xu, Zv)
            for i in range(len(D)):
                for j in range(len(D)):
                    if abs(D[i][j].real) < 1e-10:
                        D[i][j] = 0. + D[i][j].imag * 1j
                    if abs(D[i][j].imag) < 1e-10:
                        D[i][j] = D[i][j].real + 0. * 1j            

            Displacement_list.append(D)
    return Displacement_list

def Discrete_phase_point_operator_00(d1: int, d2: int) -> list[complex, complex]:
    """
    Input:   d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
    Output:   The operator A_00 
    """
    if d2 == 0:
        A00 = np.zeros((d1,d1), dtype=np.complex128)
        Displacement_operators = Displacement_Operator(d1)
        for i in range(d1):
            A00 += Displacement_operators[i]
        
        A00/= d1
    
    else:
        A00 = np.zeros((d1 * d2, d1 * d2), dtype=np.complex128)
        Displacement_operators1 = Displacement_Operator(d1)
        Displacement_operators2 = Displacement_Operator(d2)
        for i in range(len(Displacement_operators1)):
            for j in range(len(Displacement_operators2)):
                A00 += np.kron(Displacement_operators1[i], Displacement_operators2[j])
        A00 /= (d1 * d2)

    for i in range(len(A00)):
        for j in range(len(A00)):
            if abs(A00[i][j].real) < 1e-10:
                A00[i][j] = 0. + A00[i][j].imag * 1j
            if abs(A00[i][j].imag) < 1e-10:
                A00[i][j] = A00[i][j].real + 0. * 1j

    return A00

def Discrete_phase_point_operators(d1: int, d2: int) ->list[list[complex, complex]]:
    """
    Input:   d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
    Output:   The list of operators A_uv
    """
    if d2 == 0:
        #this first block calculates A00
        A00 = np.zeros((d1,d1), dtype=np.complex128)
        Displacement_operators = Displacement_Operator(d1)
        for i in range(len(Displacement_operators)):
            A00 += Displacement_operators[i]
        A00/= d1

        for i in range(len(A00)):
            for j in range(len(A00)):
                if abs(A00[i][j].real) < 1e-10:
                    A00[i][j] = 0. + A00[i][j].imag * 1j
                if abs(A00[i][j].imag) < 1e-10:
                    A00[i][j] = A00[i][j].real + 0. * 1j

        #this second block calculates Auv
        Amatrices = []
        for k in range(len(Displacement_operators)):
            aux = np.dot(A00, np.transpose(np.conjugate(Displacement_operators[k])))
            result = np.dot(Displacement_operators[k], aux)
            Amatrices.append(result)
        return Amatrices
        
    else:
        A00 = np.zeros((d1 * d2, d1 * d2), dtype=np.complex128)
        Displacement_operators1 = Displacement_Operator(d1)
        Displacement_operators2 = Displacement_Operator(d2)
        for i in range(len(Displacement_operators1)):
            for j in range(len(Displacement_operators2)):
                A00 += np.kron(Displacement_operators1[i], Displacement_operators2[j])
        A00 /= (d1 * d2)

        for i in range(len(A00)):
            for j in range(len(A00)):
                if abs(A00[i][j].real) < 1e-10:
                    A00[i][j] = 0. + A00[i][j].imag * 1j
                if abs(A00[i][j].imag) < 1e-10:
                    A00[i][j] = A00[i][j].real + 0. * 1j


        Dmatrices = [] #tensor product of the displacement operators
        for i in range(len(Displacement_operators1)):
            for j in range(len(Displacement_operators2)):
                Dmatrices.append(np.kron(Displacement_operators1[i], Displacement_operators2[j]))

        Amatrices = []
        for k in range(len(Dmatrices)):
            aux = np.dot(A00, np.transpose(np.conjugate(Dmatrices[k])))
            result = np.dot(Dmatrices[k], aux)
            Amatrices.append(result)

        return Amatrices

def Sum_Discrete_Wigner(rho: list[complex, complex], d1: int, d2: int) -> float:
    """
    Input:  rho -> density matrix 
            d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
    Output: The sum of the absolute value of the discrete Wigner function for all the possible A_uv
    """
    Amatrices = Discrete_phase_point_operators(d1, d2)
    Wigner = 0
    for i in range(len(Amatrices)):
        Wigner += np.abs(np.trace(np.dot(rho, Amatrices[i])))

    if d2 == 0:
        return Wigner / d1
    else:
        return Wigner / (d1 * d2)

def Get_Mana(rho: list[complex, complex], d1:int, d2:int) -> float:
    """
    Input:  rho -> density matrix 
            d1, d2 -> dimensions of the Hilbert spaces of the chosen particles
    Output: the observable Mana
    """
    return np.log2(Sum_Discrete_Wigner(rho, d1, d2))

def PeresHorodecki_criterion(rho: list[complex, complex], pdg_code: list[int], epsilon=1e-7) -> tuple[bool, list[float]]:
    """
    Input:  rho -> density matrix
            pdg_code -> pdg codes of the two particles in the density matrix
            epsilon -> treshold under which the eigenvalues are taken as 0 to deal with numerical instabilities
    Output: flag_entanglement -> True if the criterion is satisfied and shows entanglement
            eigvals -> eigenvalues of the partially transposed density matrix
    """
    rhoTB = Partial_Transpose(rho, 2, pdg_code)
    eigvals, _ = la.eigh(rhoTB)
    for i in range(len(eigvals)):
        if abs(eigvals[i]) < epsilon:
            eigvals[i] = 0
    flag_entanglement = False
    for eig in eigvals:
        if eig < 0.:
            flag_entanglement = True
            break

    return flag_entanglement, eigvals

def trace_distance(rho1: list[complex, complex], rho2: list[complex, complex]) -> float:
    """
       Input: rho1, rho2 -> density matrices in matrix format
       Output: trace distance between the two density matrices
    """
    aux1 = rho1 - rho2
    aux2 = np.dot(np.conjugate(np.transpose(aux1)), aux1)
    eigvals, eigvecs = la.eigh(np.array(aux2))
    for o in range(len(eigvals)): #This is necessary because numerical errors can cause a almost zero eigenvalue to be negative.
                if eigvals[o]/max(eigvals) < 1e-20:
                        eigvals[o] = 0. 
    
    sqrt_rho = np.dot(eigvecs, np.dot(np.sqrt(np.diag(eigvals)), la.inv(eigvecs)))

    return np.trace(sqrt_rho)/2.

def Fidelity(rho1: list[complex, complex], rho2: list[complex, complex]) -> float:
    """
       Input: rho1, rho2 -> density matrices
       Output: fidelity of the two density matrices
    """
    eigvals1, eigvecs1 = la.eigh(np.array(rho1))
    sqrt_rho1 = np.dot(eigvecs1, np.dot(np.sqrt(np.diag(eigvals1)), la.inv(eigvecs1)))
    aux = np.dot(sqrt_rho1, np.dot(rho2, sqrt_rho1)) #F = Tr[sqrt(aux)]
    eigvals2, _ = la.eigh(np.array(aux))

    return np.sum(np.sqrt(eigvals2)) #the trace is the sum of the eigenvalues

    
def Fidelity_distance(rho1: list[complex, complex], rho2: list[complex, complex]) -> float:
    """
       Input: rho1, rho2 -> density matrices
       Output: fidelity distance between the two density matrices
    """
    fidelity = Fidelity(rho1, rho2)
    return np.sqrt(1 - fidelity**2)


#########################END QI OBSERVABLES ############################