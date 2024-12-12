import numpy as np
import matplotlib.pyplot as plt
from assignment_3_helper import LCPSolve, assignment_3_render


# DEFINE GLOBAL PARAMETERS
L = 0.4
MU = 0.3
EP = 0.5
dt = 0.01
m = 0.3
g = np.array([0., -9.81, 0.])
rg_squared = 1./12. * (2 * L * L) #TODO: Rename this to rg_squared since it is $$r_g^2$$ - Do it also in the master
M = np.array([[m, 0, 0], [0, m, 0], [0, 0, m * rg_squared]])
Mi = np.array([[1./m, 0, 0], [0, 1./m, 0], [0, 0, 1./(m * rg_squared)]])
DELTA = 0.001
T = 150

def get_trans(q):
    '''
    Returns the 3x3 transformation matrix from world to object
    '''
    xt, yt, thetat = q[0], q[1], q[2]
    trans_mat = np.array([[np.cos(thetat), -np.sin(thetat), xt],
                          [np.sin(thetat),  np.cos(thetat), yt],
                          [0             ,  0             , 1]])
    return trans_mat

def get_corners(q):
    """
    Return a 4x2 representing the 4 corners (x,y) at current q
    3 --- 0
    |     |
    |     |
    2 --- 1
    """
    trans_mat = get_trans(q)
    corners = L / 2 * np.array([[ 1,  1, 0],
                                [ 1, -1, 0],
                                [-1,  1, 0],
                                [-1, -1, 0]])
    corners[:,-1] = 1
    corners = trans_mat @ corners.T
    return corners.T[:, :2]

def get_contacts(q):
    """
        Return jacobian of the lowest corner of the square and distance to contact
        :param q: <np.array> current configuration of the object
        :return: <np.array>, <float> jacobian and distance
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    
    corners = get_corners(q)
    phi = np.min(corners[:,1])
    contact_corner = corners[np.argmin(corners[:,1]),:]

    trans_mat = get_trans(q)
    jac = np.zeros((3,2))  
    # jac[:2,:2] = trans_mat[:2,:2].T
    jac[:2,:2] = np.array([[1, 0],
                           [0, 1]])

    vec_r = contact_corner - q[:2]
    jac[2,0] = np.cross(vec_r, np.array([1, 0]))
    jac[2,1] = np.cross(vec_r, np.array([0, 1]))
    # ------------------------------------------------
    return jac, phi


def form_lcp(jac, v):
    """
        Return LCP matrix and vector for the contact
        :param jac: <np.array> jacobian of the contact point
        :param v: <np.array> velocity of the center of mass
        :return: <np.array>, <np.array> V and p
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE
    jac_hat = np.zeros((3,3))
    jac_hat[:,0] = jac[:,1]
    jac_hat[:,1] = -jac[:,0]
    jac_hat[:,2] = jac[:,0]

    V = np.zeros((4,4))
    V[:3,:3] = jac_hat.T @ Mi @ jac_hat * dt
    V[:3, 3] = np.array([0, 1, 1])
    V[3, :] = np.array([MU, -1, -1, 0])
    
    fe = M @ g
    bt = v + Mi @ fe * dt
    p = np.zeros(4)
    p[0] = np.matmul(jac[:,1], (EP * v + bt))
    p[1] = np.matmul(-jac[:,0], bt)
    p[2] = np.matmul(jac[:,0], bt)

    # ------------------------------------------------
    return V, p


def step(q, v):
    """
        predict next config and velocity given the current values
        :param q: <np.array> current configuration of the object
        :param v: <np.array> current velocity of the object
        :return: <np.array>, <np.array> q_next and v_next
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    jac, phi = get_contacts(q)
    fe = M @ g

    if phi < DELTA:
        V, p = form_lcp(jac, v)
        f_r = lcp_solve(V, p)
        v_next =  v + dt * Mi @ (fe + jac[:,1]*f_r[0] - jac[:,0]*f_r[1] + jac[:,0]*f_r[2])
        q_next = q + dt * v_next + np.array([0, DELTA, 0])

    else:
        v_next = v + dt * Mi @ fe
        q_next = q + dt * v_next


    # ------------------------------------------------
    return q_next, v_next


def simulate(q0, v0):
    """
        predict next config and velocity given the current values
        :param q0: <np.array> initial configuration of the object
        :param v0: <np.array> initial velocity of the object
        :return: <np.array>, <np.array> q and v trajectory of the object
    """
    # ------------------------------------------------
    # FILL WITH YOUR CODE

    q = np.zeros((3, T))  # TODO: Replace with your result
    v = np.zeros((3, T))

    q[:,0] = q0
    v[:,0] = v0

    for i in range(T-1):
        q[:, i+1], v[:, i+1] = step(q[:,i], v[:,i])
    
    # ------------------------------------------------
    return q, v


def lcp_solve(V, p):
    """
        DO NOT CHANGE -- solves the LCP
        :param V: <np.array> matrix of the LCP
        :param p: <np.array> vector of the LCP
        :return: renders the trajectory
    """
    sol = LCPSolve(V, p)
    f_r = sol[1][:3]
    return f_r


def render(q):
    """
        DO NOT CHANGE -- renders the trajectory
        :param q: <np.array> configuration trajectory
        :return: renders the trajectory
    """
    assignment_3_render(q)


if __name__ == "__main__":
    # to test your final code, use the following initial configs
    q0 = np.array([0.0, 1.5, np.pi / 180. * 60.])
    # corners = get_corners(q0)
    # print(get_contacts(q0))
    # plt.scatter(corners[:,0], corners[:,1])
    # plt.show()

    v0 = np.array([0., -0.2, 0.])
    q, v = simulate(q0, v0)

    
    plt.plot(q[1, :])
    plt.show()

    render(q)




