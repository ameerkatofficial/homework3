"""
Classes to define potential and potential planner for the sphere world
"""

import numpy as np
import me570_geometry
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
        d = self.sphere.distance_grad(x_eval)
        d_i = me570_geometry.sphere.distance_influence
        if 0 < d and d < d_i: 
            grad_u_rep = -((1/d) - (1/d_i)) * (1/(d**2)) * np.gradient(d)
        elif d > d_i:
            grad_u_rep = 0
        else:
            grad_u_rep = math.nan
   
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
        x_goal = self.potential.xGoal
        if self.potential.shape == 'conic':
            p = 1
        else:
            p = 2
        u_attr = abs(x_eval - x_goal)**p
            
        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient is
    given by the formula If  potential['shape'] is equal to  'conic', use p=1; if it is equal to
    'quadratic', use p=2.
        """
        x_goal = self.potential.xGoal
        if self.potential.shape == 'conic':
            p = 1
        else:
            p = 2
        u_attr = eval(x_eval)
        grad_u_attr = p * u_attr**(p-1) * ((x_eval - x_goal)/abs(x_eval - x_goal))
        return grad_u_attr


class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """
        sum_u_rep = 0
        for sphere in self.world:
            sum_u_rep += sphere.eval(x_eval)
            
        a = self.potential.repulsiveWeight
        u_eval = self.u_attr + a * sum_u_rep
        
        return u_eval

    def grad(self, x_eval):
        
        """
        Compute the gradient of the total potential,  U= U_ attr+    _i U_ rep,i, where   is given by
    the variable  potential.repulsiveWeight
        """
        grad_u_eval = np.gradient(self.u_eval)
        return grad_u_eval


class Planner:
    """  """
    def run(self, x_start, world, potential, planned_parameters):
        """
        This function uses a given function ( planner_parameters['control']) to implement a generic
    potential-based planner with step size  planner_parameters['epsilon'], and evaluates the cost
    along the returned path. The planner must stop when either the number of steps given by
    planner_parameters['nb_steps'] is reached, or when the norm of the vector given by
    planner_parameters['control'] is less than 5 10^-3 (equivalently,  5e-3).
        """
        Planner.run_plot()
        nb_steps = Planner.planner_parameters['nb_steps']
        control = Planner.planner_parameters['control']
        epsilon = Planner.planner_parameters['epilson']
        steps = 0
        x_path = np.zeros(nb_steps, 2)
        u_path = np.zeros(nb_steps,1)
        
        x_path[0,:] = x_start
        u_path[0,:] = control.eval(x_path[0,:])
        
        while steps < nb_steps: 
            if np.linalg.norm(control) > 5e-3:
              x_path[steps + 1, :] = x_path[steps, :] + epsilon * control.grad(x_path[steps, :]) 
              u_path[steps + 1, :] = control.eval(x_path[0,:])
            else:
                break
            steps += 1
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
        world.plot()
        for i in world.x_start.shape[1]:
            for j in world.x_goal.shape[1]:
                potential = {
                    "x_goal": world.x_goal,
                    "shape":1,
                    "repulsive_weight":1}
                tot = Total(world,potential)
                #make another dictionary for potential 
                #make object tot to create potential
                planner_parameters = {
                "U" : tot.eval(x_eval),
                "control": tot.grad(x_eval) * -1,
                "epsilon": 100,
                "nb_steps": 1000
                }
            self.run(x_start, planner_parameters)
        return planner_parameters


def clfcbf_control(x_eval, world, potential):
    """
    Compute u^* according to      (  eq:clfcbf-qp  ).
    """
    pass  # Substitute with your code
    return u_opt

    