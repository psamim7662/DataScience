import numpy as np

# PSO Parameters
SWARM_SIZE = 30    # Number of particles
DIMENSIONS = 2     # Number of dimensions
MAX_ITERATIONS = 3  # Maximum number of iterations
W = 0.5    # Inertia weight
C1 = 1.5   # Cognitive coefficient
C2 = 1.5   # Social coefficient
RANGE = 5.0  # Range for initializing particle positions

# Objective function to minimize
def objective_function(x):
    return np.sum(x ** 2)

def pso_algorithm():
    # Initialize particles
    swarm = np.random.uniform(-RANGE, RANGE, (SWARM_SIZE, DIMENSIONS))
    velocities = np.random.uniform(-RANGE, RANGE, (SWARM_SIZE, DIMENSIONS))
    personal_best_positions = np.copy(swarm)
    personal_best_values = np.apply_along_axis(objective_function, 1, swarm)
    
    global_best_index = np.argmin(personal_best_values)
    global_best_position = personal_best_positions[global_best_index]
    global_best_value = personal_best_values[global_best_index]
    
    for t in range(MAX_ITERATIONS):
        # Update velocities and positions
        r1, r2 = np.random.rand(SWARM_SIZE, DIMENSIONS), np.random.rand(SWARM_SIZE, DIMENSIONS)
        velocities = (W * velocities +
                      C1 * r1 * (personal_best_positions - swarm) +
                      C2 * r2 * (global_best_position - swarm))
        swarm += velocities
        
        # Evaluate fitness
        fitness_values = np.apply_along_axis(objective_function, 1, swarm)
        
        # Update personal bests
        better_mask = fitness_values < personal_best_values
        personal_best_positions[better_mask] = swarm[better_mask]
        personal_best_values[better_mask] = fitness_values[better_mask]
        
        # Update global best
        current_best_index = np.argmin(personal_best_values)
        current_best_value = personal_best_values[current_best_index]
        if current_best_value < global_best_value:
            global_best_position = personal_best_positions[current_best_index]
            global_best_value = current_best_value

        # Print local and global bests for this iteration
        print(f"Iteration {t+1}")
        print("  Local Best Positions and Values:")
        for i in range(SWARM_SIZE):
            print(f"    Particle {i}: Position {personal_best_positions[i]}, Value {personal_best_values[i]}")
        print("  Global Best Position and Value:")
        print(f"    Global Best Position: {global_best_position}, Global Best Value: {global_best_value}\n")
        
    return global_best_position, global_best_value

if __name__ == "__main__":
    best_position, best_value = pso_algorithm()
    print("Final Best Position:", best_position)
    print("Final Best Value:", best_value)






# import numpy as np
# import matplotlib.pyplot as plt

# class PSO(object):
    
    
     
# #         """
# #            Class implementing PSO algorithm.
# #        """
#     def __init__(self, func, init_pos, n_particles):
        
        
# #           """
# #              Initialize the key variables.
# #                 Args:
# #                   func (function): the fitness function to optimize.
# #                  init_pos (array-like): the initial position to kick off the
# #                                optimization process.
# #                    n_particles (int): the number of particles of the swarm.
# #                  """
#         self.func = func
#         self.n_particles = n_particles
#         self.init_pos = np.array(init_pos)
#         self.particle_dim = len(init_pos)
#         # Initialize particle positions using a uniform distribution
#         self.particles_pos = np.random.uniform(size=(n_particles, self.particle_dim)) \
#                         * self.init_pos
#         # Initialize particle velocities using a uniform distribution
#         self.velocities = np.random.uniform(size=(n_particles, self.particle_dim))

#         # Initialize the best positions
#         self.g_best = init_pos
#         self.p_best = self.particles_pos
    
    


#     def update_position(self, x, v):
        
# #     """
# #       Update particle position.
# #       Args:
# #         x (array-like): particle current position.
# #         v (array-like): particle current velocity.
# #       Returns:
# #         The updated position (array-like).
# #     """
#         x = np.array(x)
#         v = np.array(v)
#         new_x = x + v
#         return new_x

#     def update_velocity(self, x, v, p_best, g_best, c0=0.5, c1=1.5, w=0.75):
        
# #     """
# #       Update particle velocity.
# #       Args:
# #         x (array-like): particle current position.
# #         v (array-like): particle current velocity.
# #         p_best (array-like): the best position found so far for a particle.
# #         g_best (array-like): the best position regarding
# #                              all the particles found so far.
# #         c0 (float): the cognitive scaling constant.
# #         c1 (float): the social scaling constant.
# #         w (float): the inertia weight
# #       Returns:
# #         The updated velocity (array-like).
# #     """
#         x = np.array(x)
#         v = np.array(v)
#         assert x.shape == v.shape, 'Position and velocity must have same shape'
#         # a random number between 0 and 1.
#         r = np.random.uniform()
#         p_best = np.array(p_best)
#         g_best = np.array(g_best)

#         new_v = w*v + c0 * r * (p_best - x) + c1 * r * (g_best - x)
#         return new_v

#     def optimize(self, maxiter=200):
        
# #     """
# #       Run the PSO optimization process untill the stoping criteria is met.
# #       Case for minimization. The aim is to minimize the cost function.
# #       Args:
# #           maxiter (int): the maximum number of iterations before stopping
# #                          the optimization.
# #       Returns:
# #           The best solution found (array-like).
# #     """
#         for k in range(0,maxiter):
#               print('global best in iter: ', k, 'is',self.g_best)
        
        
            
#               for i in range(self.n_particles):
                    
                    
#                     x = self.particles_pos[i]
#                     v = self.velocities[i]
#                     p_best = self.p_best[i]
#                     self.velocities[i] = self.update_velocity(x, v, p_best, self.g_best)
#                     self.particles_pos[i] = self.update_position(x, v)
#                     # Update the best position for particle i
#                     if self.func(self.particles_pos[i]) < self.func(p_best):
                        
#                         self.p_best[i] = self.particles_pos[i]
#                     # Update the best position overall
#                     if self.func(self.particles_pos[i]) < self.func(self.g_best):
                            
#                         self.g_best = self.particles_pos[i]
                    
#         return self.g_best, self.func(self.g_best)
# # Example of the sphere function
# def sphere(x):
# #   """
# #     In 3D: f(x,y,z) = x² + y² + z²
# #   """
#   return np.sum(np.square(x))

# if __name__ == '__main__':
    
#     init_pos = [2,1,3]
#     PSO_s = PSO(func=sphere, init_pos=init_pos, n_particles=50)
#     res_s = PSO_s.optimize()
#     print("Sphere function")
#     print(f'x = {res_s[0]}') # x = [-0.00025538 -0.00137996  0.00248555]
#     print(f'f = {res_s[1]}') # f = 8.14748063004205e-06

