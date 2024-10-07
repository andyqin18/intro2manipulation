import numpy as np
import cvxpy as cp
np.set_printoptions(precision=3, suppress=True)


def get_contacts():
    """
        Return contact normals and locations as a matrix
        :return:
            - Contact Matrix R: <np.array> of size (2,3) containing the contact locations [r0 | r1 | r2]
            - Normal Matrix N: <np.array> of size (2,3) containing the contact locations [n0 | n1 | n2]
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    R = np.array([[0.5/(2**0.5), -0.5, 0.0],
                  [0.5/(2**0.5),  0.0, -0.5]])
    N = np.array([[-1/(2**0.5), 1.0, 0.0],
                  [-1/(2**0.5), 0.0, 1.0]])
    # ------------------------------------------------
    return R, N


def calculate_grasp(R, N):
    """
        Return the grasp matrix as a function of contact locations and normals
        :param R: <np.array> locations of contact
        :param N: <np.array> contact normals
        :return: <np.array> of size (3,6) Grasp matrix for Fig. 1 containing [ J0 | J1 | J2]
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    G = np.zeros((3, 6))
    for i in range(3):
        nx, ny, rx, ry = N[0,i], N[1,i], R[0,i], R[1,i]
        G[:,2*i] = np.array([ny, -nx, -rx*nx-ry*ny])
        G[:,2*i+1] = np.array([nx, ny, rx*ny-ry*nx])

    # ------------------------------------------------
    return G


def calculate_facet(mu):
    """
        Return friction cone representation in terms of facet normals
        :param mu: <float> coefficient of friction
        :return: <np.array> Facet normal matrix
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    F_i = np.array([[1, mu],
                    [-1, mu]]) / (1 + mu**2)**0.5
    F = np.kron(np.eye(3), F_i)
    # ------------------------------------------------
    return F


def compute_grasp_rank(G):
    """
        Return boolean of if grasp has rank 3 or not
        :param G: <np.array> grasp matrix as a numpy array
        :return: <bool> boolean flag for if rank is 3 or not
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    flag = np.linalg.matrix_rank(G) == 3
    # ------------------------------------------------
    return flag


def compute_constraints(G, F):
    """
        Return grasp constraints as numpy arrays
        :param G: <np.array> grasp matrix as a numpy array
        :param F: <np.array> friction cone facet matrix as a numpy array
        :return: <np.array>x5 contact constraints
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    A = np.hstack([G, np.zeros((3,1))])
    b = np.zeros(3)  

    P = np.zeros((8,7))
    P[:6,:6] = -F                       # -Ffc + d <= 0
    P[:6, 6] = np.ones(6)
    P[-2,-1] = -1                       # -d <= 0
    P[-1,:6] = np.array([0,1,0,1,0,1])  # eTfc <= nc

    q = np.zeros(8)
    q[-1] = 3

    c = np.zeros((7,1))
    c[-1] = 1
    # ------------------------------------------------
    return A, b, P, q, c


def check_force_closure(A, b, P, q, c):
    """
        Solves Linear program given grasp constraints - DO NOT EDIT
        :return: d_star
    """
    # ------------------------------------------------
    # DO NOT EDIT THE CODE IN THIS FUNCTION
    x = cp.Variable(A.shape[1])

    prob = cp.Problem(cp.Maximize(c.T@x),
                      [P @ x <= q, A @ x == b])
    prob.solve()
    d = prob.value
    print('Optimal value of d (d^*): {:3.2f}'.format(d))
    return d
    # ------------------------------------------------


if __name__ == "__main__":
    mu = 0.3
    R, N = get_contacts()

    F = calculate_facet(mu=mu)
    G = calculate_grasp(R,N)
    d = check_force_closure(*compute_constraints(G,F))
    print(d)


