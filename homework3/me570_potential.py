"""
Classes to define potential and potential planner for the sphere world
"""

import numpy as np
import me570_geometry
import me570_qp
from matplotlib import pyplot as plt
from scipy import io as scio
import math


class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different goal locations (one
    for each column).
        """
        data = scio.loadmat('sphereWorld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = np.asscalar(sphere_args[1])
            sphere_args[2] = np.asscalar(sphere_args[2])
            self.world.append(me570_geometry.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def plot(self):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a  * marker at the goal location.
        """

        for sphere in self.world:
            sphere.plot('r')

        plt.scatter(self.x_goal[0, :], self.x_goal[1, :], c='g', marker='*')

        plt.xlim([-11, 11])
        plt.ylim([-11, 11])


class RepulsiveSphere:
    """ Repulsive potential for a sphere """
    def __init__(self, sphere):
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """
        Evaluate the repulsive potential from  sphere at the location x= x_eval. The function returns
    the repulsive potential as given by      (  eq:repulsive  ).
        """
        d = self.sphere.distance(x_eval)
        d_i = self.sphere.distance_influence
        if 0 < d and d < d_i: 
            u_rep = 0.5 * ((1/d) - (1/d_i))**2 
        elif d > d_i:
            u_rep = 0
        else:
            u_rep = math.nan
            
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by      (  eq:repulsive-gradient
    ).
        """
        d = self.sphere.distance(x_eval)
        d_grad = self.sphere.distance_grad(x_eval)
        d_i = self.sphere.distance_influence
        if 0 < d and d < d_i: 
            grad_u_rep = -((1/d) - (1/d_i)) * (1/(d**2)) * d_grad
        elif d > d_i:
            grad_u_rep = np.zeros((2,1))
        else:
            grad_u_rep = np.nan*np.ones((2,1))
   
        return grad_u_rep


class Attractive:
    """ Repulsive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        x_goal = self.potential["x_goal"]
        if self.potential["shape"] == 'conic':
            p = 1
        else:
            p = 2
        u_attr = np.linalg.norm(x_eval - x_goal)**p
            
        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient is
    given by the formula If  potential['shape'] is equal to  'conic', use p=1; if it is equal to
    'quadratic', use p=2.
        """
        x_goal = self.potential["x_goal"]
        if self.potential["shape"] == 'conic':
            p = 1
        else:
            p = 2
        grad_u_attr = p * np.linalg.norm(x_eval - x_goal)**(p-2) * (x_eval - x_goal)
        return grad_u_attr


class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential
        self.u_attr = Attractive(potential)
        self.u_rep = list(map(lambda x: RepulsiveSphere(x), world.world))

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """
        sum_u_rep = 0
        for sphere in self.u_rep:
            sum_u_rep += sphere.eval(x_eval)
            
        a = self.potential["repulsive_weight"]
        u_eval = self.u_attr.eval(x_eval) + a * sum_u_rep
        
        return u_eval

    def grad(self, x_eval):
        
        """
        Compute the gradient of the total potential,  U= U_ attr+    _i U_ rep,i, where   is given by
    the variable  potential.repulsiveWeight
        """
        sum_u_rep = 0
        for sphere in self.u_rep:
            sum_u_rep += sphere.grad(x_eval)
            
        a = self.potential["repulsive_weight"]
        grad_u_eval = self.u_attr.grad(x_eval) + a * sum_u_rep 
        
        return grad_u_eval


class Planner:
    """  """
    def run(self, x_start, planner_parameters):
        """
        This function uses a given function ( planner_parameters['control']) to implement a generic
    potential-based planner with step size  planner_parameters['epsilon'], and evaluates the cost
    along the returned path. The planner must stop when either the number of steps given by
    planner_parameters['nb_steps'] is reached, or when the norm of the vector given by
    planner_parameters['control'] is less than 5 10^-3 (equivalently,  5e-3).
        """
        nb_steps = planner_parameters['nb_steps']
        pot_func = planner_parameters['U']
        control = planner_parameters['control']
        epsilon = planner_parameters['epsilon']
        
        x_path = np.zeros((2,nb_steps))
        u_path = np.zeros((1,nb_steps))
        
        print(x_start.shape)
        x_path[:,0] = x_start.T
        u_path[:,0] = pot_func(np.reshape(x_path[:,0],(2,1)))

        x_path[:] = np.nan
        u_path[:] = np.nan
        
        x_path = [x_start] 
        u_path = [pot_func(x_path[-1])]
        
        for iter in range(1, nb_steps):
            grad = control(x_path[-1])
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 5e-3:
                x_path.append(np.array([[np.nan], [np.nan]]))
                u_path.append(np.nan)
                continue
            step = epsilon * grad
            x_path.append(x_path[-1] - step)
            u_path.append(pot_func(x_path[-1]))
            
        x_path = np.hstack(x_path)
        u_path = np.array(u_path)
        return x_path, u_path
        


    def run_plot(self):
            """
            This function performs the following steps:
         - Loads the problem data from the file !70!DarkSeaGreen2 sphereworld.mat.
         - For each goal location in  world.xGoal:
         - Uses the function Sphereworld.plot to plot the world in a first figure.
         - Sets  planner_parameters['U'] to the negative of  Total.grad.
         - it:grad-handle Calls the function Potential.planner with the problem data and the input
        arguments. The function needs to be called five times, using each one of the initial locations
        given in  x_start (also provided in !70!DarkSeaGreen2 sphereworld.mat).
         - it:plot-plan After each call, plot the resulting trajectory superimposed to the world in the
        first subplot; in a second subplot, show  u_path (using the same color and using the  semilogy
        command).
            """
            world = SphereWorld()
            for j in range(world.x_goal.shape[1]):
                for i in range(world.x_start.shape[1]):
                    potential = {
                        "x_goal": world.x_goal[:,j].reshape((2, 1)),
                        "shape":"quadratic",
                        "repulsive_weight":1}
                    att = Attractive(potential)
                    att_eval = lambda x_eval: att.eval(x_eval)
                    tot = Total(world,potential)
                    tot_grad = lambda x_eval: tot.grad(x_eval) 
                    tot_eval = lambda x_eval: tot.eval(x_eval)
                    clfcbf_grad = lambda x_eval: clfcbf_control(x_eval, world, potential)
                    planner_parameters = {
                    # "U" : tot_eval,
                    # "control": tot_grad,
                    "U": att_eval,
                    "control": clfcbf_grad,
                    "epsilon": 0.1,
                    "nb_steps": 40
                    }
                    x_path, u_path = self.run(world.x_start[:,i].reshape((2, 1)), planner_parameters)
                    print(x_path)
                    world.plot()
                    plt.plot(x_path[0, :], x_path[1, :])  
                plt.show()
                
                 
            return planner_parameters
def clfcbf_control(x_eval, world, potential):
    """
    Compute u^* according to      (  eq:clfcbf-qp  ).
    """
    a_barrier = np.zeros((len(world.world), 2))
    b_barrier= np.zeros((len(world.world), 1))
    attractive = Attractive(potential)
    u_ref = attractive.grad(x_eval)
    row = 0
    c_h = potential['repulsive_weight']
    for sphere in world.world:
        h = sphere.distance(x_eval)
        h_grad = sphere.distance_grad(x_eval)
        a_barrier[row,:] = h_grad.T
        b_barrier[row,:] = h
        row+=1
    b_barrier*= -c_h    
    print(a_barrier)
    print()
    print(b_barrier)
    print()
    print(u_ref)
    print()
    
        
    #feed np arrays of a_barrier, b_barrier, and u_ref int me570_qp
    #u ref, clubs is just the function already written 
    u_opt = me570_qp.qp_supervisor(a_barrier, b_barrier, u_ref)
    
    return u_opt
    


if __name__=="__main__":

     p = Planner() 
     p.run_plot()
     



    