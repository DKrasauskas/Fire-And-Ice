# General imports
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Tudat imports
import tudatpy
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.dynamics import environment_setup
from tudatpy.util import result2array
from tudatpy.astro.time_representation import DateTime

# Pygmo imports
import pygmo as pg


def convert_trajectory_parameters(
        transfer_trajectory_object: tudatpy.trajectory_design.transfer_trajectory.TransferTrajectory,
        trajectory_parameters: List[float],
) -> Tuple[List[float], List[List[float]], List[List[float]]]:
    # Declare lists of transfer parameters
    node_times = list()
    leg_free_parameters = list()
    node_free_parameters = list()

    # Extract from trajectory parameters the lists with each type of parameters
    departure_time = trajectory_parameters[0]
    times_of_flight_per_leg = trajectory_parameters[1:]

    # Get node times
    # Node time for the initial node: departure time
    node_times.append(departure_time)
    # None times for other nodes: node time of the previous node plus time of flight
    accumulated_time = departure_time
    for i in range(0, transfer_trajectory_object.number_of_nodes - 1):
        accumulated_time += times_of_flight_per_leg[i]
        node_times.append(accumulated_time)

    # Get leg free parameters and node free parameters: one empty list per leg
    for i in range(transfer_trajectory_object.number_of_legs):
        leg_free_parameters.append([])
    # One empty array for each node
    for i in range(transfer_trajectory_object.number_of_nodes):
        node_free_parameters.append([])

    return node_times, leg_free_parameters, node_free_parameters


###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################


class TransferTrajectoryProblem:

    def __init__(
            self,
            transfer_trajectory_object: tudatpy.trajectory_design.transfer_trajectory.TransferTrajectory,
            departure_date_lb: float,  # Lower bound on departure date
            departure_date_ub: float,  # Upper bound on departure date
            legs_tof_lb: np.ndarray,  # Lower bounds of each leg's time of flight
            legs_tof_ub: np.ndarray,
    ):  # Upper bounds of each leg's time of flight
        """
        Class constructor.
        """

        self.departure_date_lb = departure_date_lb
        self.departure_date_ub = departure_date_ub
        self.legs_tof_lb = legs_tof_lb
        self.legs_tof_ub = legs_tof_ub

        # Save the transfer trajectory object as a lambda function
        # PyGMO internally pickles its user defined objects and some objects cannot be pickled properly without using lambda functions.
        self.transfer_trajectory_function = lambda: transfer_trajectory_object

    def get_bounds(self) -> tuple:
        """
        Returns the boundaries of the decision variables.
        """

        # Retrieve transfer trajectory object
        transfer_trajectory_obj = self.transfer_trajectory_function()

        number_of_parameters = self.get_number_of_parameters()

        # Define lists to save lower and upper bounds of design parameters
        lower_bound = list(np.empty(number_of_parameters))
        upper_bound = list(np.empty(number_of_parameters))

        # Define boundaries on departure date
        lower_bound[0] = self.departure_date_lb
        upper_bound[0] = self.departure_date_ub

        # Define boundaries on time of flight between bodies ['Earth', 'Venus', 'Venus', 'Earth', 'Jupiter', 'Saturn']
        for i in range(0, transfer_trajectory_obj.number_of_legs):
            lower_bound[i + 1] = self.legs_tof_lb[i]
            upper_bound[i + 1] = self.legs_tof_ub[i]

        bounds = (lower_bound, upper_bound)
        return bounds

    def get_number_of_parameters(self):
        """
        Returns number of parameters that will be optimized
        """

        # Retrieve transfer trajectory object
        transfer_trajectory_obj = self.transfer_trajectory_function()

        # Get number of parameters: it's the number of nodes (time at the first node, and time of flight to reach each subsequent node)
        number_of_parameters = transfer_trajectory_obj.number_of_nodes

        return number_of_parameters

    def fitness(self, trajectory_parameters: List[float]) -> list:
        """
        Returns delta V of the transfer trajectory object with the given set of trajectory parameters
        """

        # Retrieve transfer trajectory object
        transfer_trajectory = self.transfer_trajectory_function()

        # Convert list of trajectory parameters to appropriate format
        node_times, leg_free_parameters, node_free_parameters = (
            convert_trajectory_parameters(transfer_trajectory, trajectory_parameters)
        )

        # Evaluate trajectory
        try:
            transfer_trajectory.evaluate(
                node_times, leg_free_parameters, node_free_parameters
            )
            delta_v = transfer_trajectory.delta_v

        # If there was some error in the evaluation of the trajectory, use a very large deltaV as penalty
        except:
            delta_v = 1e10

        return [delta_v]


###########################################################################
# Define transfer trajectory properties
###########################################################################

# Define the central body
central_body = "Sun"

# Define order of bodies (nodes)
transfer_body_order = ["Earth", "Venus", "Earth", "Jupiter"]

# Define departure orbit
departure_semi_major_axis = np.inf
departure_eccentricity = 0

# Define insertion orbit
arrival_semi_major_axis = 1.0895e8 / 0.02
arrival_eccentricity = 0.98

# Create simplified system of bodies
bodies = environment_setup.create_simplified_system_of_bodies()

# Define the trajectory settings for both the legs and at the nodes
transfer_leg_settings, transfer_node_settings = (
    transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
        transfer_body_order,
        departure_orbit=(departure_semi_major_axis, departure_eccentricity),
        arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
    )
)

# Create the transfer calculation object
transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body,
)

# Lower and upper bound on departure date
departure_date_lb = DateTime(2020, 4, 6).to_epoch()
departure_date_ub = DateTime(2030, 12, 31).to_epoch()

# List of lower and upper on time of flight for each leg
legs_tof_lb = np.zeros(5)
legs_tof_ub = np.zeros(5)
# EE
legs_tof_lb[0] = 30 * constants.JULIAN_DAY
legs_tof_ub[0] = 300 * constants.JULIAN_DAY
# EV
legs_tof_lb[0] = 30 * constants.JULIAN_DAY
legs_tof_ub[0] = 300 * constants.JULIAN_DAY
# EV
legs_tof_lb[1] = 20 * constants.JULIAN_DAY
legs_tof_ub[1] = 600 * constants.JULIAN_DAY
# VE
legs_tof_lb[2] = 220 * constants.JULIAN_DAY
legs_tof_ub[2] = 1500 * constants.JULIAN_DAY

###########################################################################
# Setup optimization
###########################################################################
# Initialize optimization class
optimizer = TransferTrajectoryProblem(
    transfer_trajectory_object,
    departure_date_lb,
    departure_date_ub,
    legs_tof_lb,
    legs_tof_ub,
)

# Creation of the pygmo problem object
prob = pg.problem(optimizer)

# To print the problem's information: uncomment the next line
# print(prob)

# Define number of generations per evolution
number_of_generations = 10

# Fix seed
optimization_seed = 4444

# Create pygmo algorithm object
algo = pg.algorithm(pg.de(gen=number_of_generations, seed=optimization_seed, F=0.5))

# To print the algorithm's information: uncomment the next line
# print(algo)

# Set population size
population_size = 20

# Create population
pop = pg.population(prob, size=population_size, seed=optimization_seed)

###########################################################################
# Run optimization
###########################################################################

# Set number of evolutions
number_of_evolutions = 800

# Initialize empty containers
individuals_list = []
fitness_list = []

for i in range(number_of_evolutions):
    pop = algo.evolve(pop)

    # individuals save
    individuals_list.append(pop.champion_x)
    fitness_list.append(pop.champion_f)

print("The optimization has finished")

###########################################################################
# Results post-processing
###########################################################################

# Extract the best individual
print("\n########### CHAMPION INDIVIDUAL ###########\n")
print("Total Delta V [m/s]: ", pop.champion_f[0])
best_decision_variables = pop.champion_x / constants.JULIAN_DAY
print("Departure time w.r.t J2000 [days]: ", best_decision_variables[0])
print("Earth-Mars time of flight [days]: ", best_decision_variables[1])
print("Mars-Earth time of flight [days]: ", best_decision_variables[2])
print("EArth-Jupiter time of flight [days]: ", best_decision_variables[3])

# Plot fitness over generations
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(
    np.arange(0, number_of_evolutions),
    np.float_(fitness_list) / 1000,
    label="Function value: Feval",
)
# Plot champion
champion_n = np.argmin(np.array(fitness_list))
ax.scatter(
    champion_n,
    np.min(fitness_list) / 1000,
    marker="x",
    color="r",
    label="All-time champion",
    zorder=10,
)

# Prettify
ax.set_xlim((0, number_of_evolutions))
ax.set_ylim([4, 25])
ax.grid("major")
ax.set_title("Best individual over generations", fontweight="bold")
ax.set_xlabel("Number of generation")
ax.set_ylabel(r"$\Delta V [km/s]$")
ax.legend(loc="upper right")
plt.tight_layout()
plt.legend()

# Reevaluate the transfer trajectory using the champion design variables
node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(
    transfer_trajectory_object, pop.champion_x
)
transfer_trajectory_object.evaluate(
    node_times, leg_free_parameters, node_free_parameters
)

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

# Plot the state history
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au)
ax.scatter(
    fly_by_states[0, 0] / au,
    fly_by_states[0, 1] / au,
    color="blue",
    label="Earth departure",
)
ax.scatter(
    fly_by_states[1, 0] / au,
    fly_by_states[1, 1] / au,
    color="green",
    label="Mars Flyby",
)
ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, color="green")
ax.scatter(
    fly_by_states[2, 0] / au,
    fly_by_states[2, 1] / au,
    color="brown",
    label="Earth fly-by",
)

ax.scatter(
    fly_by_states[3, 0] / au,
    fly_by_states[3, 1] / au,
    color="grey",
    label="Jupiter arrival",
)
ax.scatter([0], [0], color="orange", label="Sun")
ax.set_xlabel("x wrt Sun [AU]")
ax.set_ylabel("y wrt Sun [AU]")
ax.set_aspect("equal")
ax.legend(bbox_to_anchor=[1, 1])
plt.show()