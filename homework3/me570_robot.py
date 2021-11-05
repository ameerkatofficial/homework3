import numpy as np
import me570_potential as pot
import matplotlib.pyplot as plt
import me570_geometry_hw2 as gm


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
        self.world = world
        # self.potential = potential 
        # self.planner = pot.Planner()
        # self.plannerParameters = pot.Planner.planner_parameters()
        self.total = pot.Total(world, potential)
        self.twolink = TwoLink()

    def eval(self, theta_eval):
        """
        Compute the potential U pulled back through the kinematic map of the two-link manipulator, i.e.,
    U(  Wp_ eff(  )), where U is defined as in Question~ q:total-potential, and   Wp_ eff( ) is the
    position of the end effector in the world frame as a function of the joint angles   = _1\\ _2.
        """
        vertex_effector_transf, polygon1_transf, polygon2_transf = self.twolink.kinematic_map(theta_eval)
        
        u_eval_theta = self.total.eval(vertex_effector_transf)
        
        return u_eval_theta

    def grad(self, theta_eval):
        """
        Compute the gradient of the potential U pulled back through the kinematic map of the two-link
    manipulator, i.e.,  _   U(  Wp_ eff(  )).
        """
        jacobian = self.twolink.jacobian_matrix(theta_eval)
        
        vertex_effector_transf, polygon1_transf, polygon2_transf = self.twolink.kinematic_map(theta_eval)
         
        grad_u_eval = self.total.grad(vertex_effector_transf)
        
        grad_u_eval_theta = grad_u_eval @ jacobian
        
        return grad_u_eval_theta
    
    def run(self, theta_start, planner_parameters):
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
        
        theta_path = np.zeros((2,nb_steps))
        u_path = np.zeros((1,nb_steps))
        
        print(theta_start.shape)
        theta_path[:,0] = theta_start.T
        u_path[:,0] = pot_func(np.reshape(theta_path[:,0],(2,1)))

        theta_path[:] = np.nan
        u_path[:] = np.nan
        
        theta_path = [theta_start] 
        u_path = [pot_func(theta_path[-1])]
        
        for iter in range(1, nb_steps):
            if np.isnan(theta_path[-1]).any():
                break
            grad = control(theta_path[-1])
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 5e-3:
                theta_path.append(np.array([[np.nan], [np.nan]]))
                u_path.append(np.nan)
                continue
            step = epsilon * grad
            theta_path.append(theta_path[-1] - step)
            u_path.append(pot_func(theta_path[-1]))
            
        theta_path = np.hstack(theta_path)
        u_path = np.array(u_path)
        return theta_path, u_path
  
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
        print(self.world.x_goal)
        # for j in range(self.world.x_goal[1,:].shape[1]):
        for i in range(self.world.theta_start.shape[1]):
            # potential = {
            #     "theta_goal": self.world.theta_goal[:,j].reshape((2, 1)),
            #     "shape":"quadratic",
            #     "repulsive_weight":1/4}
            grad = lambda theta_eval: self.grad(theta_eval)
            total = lambda theta_eval: self.eval(theta_eval)
            
            plannerParameters['U'] = total
            plannerParameters['control'] = grad 
            x_path, u_path = self.run(self.world.theta_start[:,i].reshape((2, 1)), plannerParameters)
            print(x_path)
            world.plot()
            plt.plot(x_path[:,1:5:-1])  
        plt.show()
        

if __name__=="__main__":
     planner = pot.Planner()
     world = pot.SphereWorld()
     plannerParameters = {"U" : None,
                          "control": None,
                          "epsilon": 0.01,
                          "nb_steps": 1000}
     potential = {
    "x_goal": world.x_goal[:,1].reshape((2, 1)),
    "shape":"quadratic",
    "repulsive_weight":1/4}
     # plannerParameters = planner.run_plot()
     t = TwoLinkPotential(world, potential) 
     t.run_plot(plannerParameters)
        
        
