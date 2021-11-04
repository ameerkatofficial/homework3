import numpy as np
import me570_geometry as gm
import me570_potential as pot
import matplotlib.pyplot as plt


class TwoLink:
    """ See description from previous homework assignments. """
    def __init__(self):
        add_y_reflection = lambda vertices: np.hstack(
            [vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])

        vertices1 = np.array([[0, 5], [-1.11, -0.511]])
        vertices1 = add_y_reflection(vertices1)
        vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                              [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
        vertices2 = add_y_reflection(vertices2)
        self.Polygons = (gm.Polygon(vertices1), gm.Polygon(vertices2))
        
    def polygons(self):
        """
        Returns two polygons that represent the links in a simple 2-D two-link manipulator.
        """
        return self.Polygons
    
    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the vertices of the links, all
    transformed according to  _1, _2.
        """
        #VARIABLES
        poly1 = self.polygons()[0]
        poly2 = self.polygons()[1]
        P_effList = [5,0]
        P_eff = np.array(P_effList)
        r_matrix_1 = gm.rot2d(theta[0])
        r_matrix_2 = gm.rot2d(theta[1])
        
        polygon1_transf = np.dot(r_matrix_1, poly1.vertices)
        
        polygon2_transf = np.dot(r_matrix_2, poly2.vertices) + np.dot(r_matrix_1, poly2.vertices)
        
        vertex_effector_transf = np.dot(r_matrix_2, P_eff) + np.dot(r_matrix_1, P_eff)
    
        return vertex_effector_transf, polygon1_transf, polygon2_transf
    
    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        [vertex_effector_transf, polygon1_transf,
         polygon2_transf] = self.kinematic_map(theta)
        plt.plot(polygon1_transf, color)
        plt.plot(polygon2_transf, color)
        
    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """
        flag_theta = []
        
        polygon_transf = self.kinematic_map(theta)
        polygon1_transf = polygon_transf[1]
        polygon2_transf = polygon_transf[2]
        Polygons_transformed = (gm.Polygon(polygon1_transf), gm.Polygon(polygon2_transf))
        if Polygons_transformed[0].is_collision(points) == True or Polygons_transformed[1].is_collision(points) == True:
            flag_theta.append(True)
        else:
            flag_theta.append(False)
    
        return flag_theta
    
    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.
        """
        #load list of bools from is_collision
        bool_list = self.is_collision(theta, points)
        #see if bool_list has TRUE in it for a certain config
        if True in bool_list:
        #if True is in bool_list, plot manipulator as red
            self.plot(theta = theta, color = 'r')
        #else, plot manipulator as green
        else:
            self.plot(theta = theta, color = 'g')
        #plot points as black astericks 
        ax = plt.gca()
        ax.scatter(points[0,:],points[1,:], color = 'k', marker = '$*$')
        plt.show()
        
    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """

        vertex_effector_dot = []
        for i in range(theta.shape[1]):
            vertex_effector_dot.append(np.array([
                [-5*np.sin(theta[0][i])*theta_dot[0][i] - 5*np.sin(theta[0][i] + theta[1][i])*(theta_dot[0][i] + theta_dot[1][i])],
                [ 5*np.cos(theta[0][i])*theta_dot[0][i] + 5*np.cos(theta[0][i] + theta[1][i])*(theta_dot[0][i] + theta_dot[1][i])]
            ]))

        return np.hstack(vertex_effector_dot)
        
    def jacobian_matrix(self, theta):
        """
        Compute the matrix representation of the Jacobian of the position of the end effector with
    respect to the joint angles as derived in Question~ q:jacobian-matrix.
        """
        
        jtheta = np.zeros(2,2)
        theta1 = theta(0,0)
        theta2 = theta(1,0)
        jtheta = np.array([[-5*np.sin(theta1 + theta2) - 5*np.sin(theta1), -5*np.sin(theta1 + theta2)],[5*np.cos(theta1 + theta2) + 5*np.cos(theta1), 5*np.cos(theta1 + theta2)]])
        
        return jtheta


class TwoLinkPotential:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        world = pot.SphereWorld()
        potential = pot.Planner.potential()

    def eval(self, theta_eval):
        """
        Compute the potential U pulled back through the kinematic map of the two-link manipulator, i.e.,
    U(  Wp_ eff(  )), where U is defined as in Question~ q:total-potential, and   Wp_ eff( ) is the
    position of the end effector in the world frame as a function of the joint angles   = _1\\ _2.
        """
        pass  # Substitute with your code
        return u_eval_theta

    def grad(self, theta_eval):
        """
        Compute the gradient of the potential U pulled back through the kinematic map of the two-link
    manipulator, i.e.,  _   U(  Wp_ eff(  )).
        """
        pass  # Substitute with your code
        return grad_u_eval_theta

    def run_plot(self, plannerParameters):
        """
        This function performs the same steps as Planner.run_plot in Question~ q:potentialPlannerTest,
    except for the following:
     - In step  it:grad-handle:  planner_parameters['U'] should be set to  @twolink_total, and
    planner_parameters['control'] to the negative of  @twolink_totalGrad.
     - In step  it:grad-handle: Use the contents of the variable  thetaStart instead of  xStart to
    initialize the planner, and use only the second goal  x_goal[:,1].
     - In step  it:plot-plan: Use Twolink.plotAnimate to plot a decimated version of the results of
    the planner. Note that the output  xPath from Potential.planner will really contain a sequence
    of join angles, rather than a sequence of 2-D points. Plot only every 5th or 10th column of
    xPath (e.g., use  xPath(:,1:5:end)). To avoid clutter, plot a different figure for each start.
        """
        print("hello world")
        world = self.world
        for j in range(world.x_goal[:,1].shape[1]):
            for i in range(world.theta_start.shape[1]):
                   potential = {
                       "x_goal": world.x_goal[:,1].reshape((2, 1)),
                       "shape":"quadratic",
                       "repulsive_weight":0.1}
                   tot = pot.Total(world,potential)
                   tot_grad = lambda x_eval: tot.grad(x_eval) 
                   tot_eval = lambda x_eval: tot.eval(x_eval)
                   clfcbf_control = lambda x_eval: clfcbf_control(x_eval, world, potential)
                   planner_parameters = {
                       "U":TwoLinkPotential.eval(theta_eval),
                       "control": TwoLinkPotential.grad(theta_eval),
                       "epsilon": 0.01,
                       "nb_steps": 10000
                   }
                   x_path, u_path = self.run(world.theta_start[:,i].reshape((2, 1)), planner_parameters)
                   print(x_path)
                   world.plot()
                   plt.plot(x_path[0, :], x_path[1, :])  
        plt.show()
