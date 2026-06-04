from injection.mgas.main import*

###########################################################################
# Setup optimization
###########################################################################
# Initialize optimization class


def optimize(transfer_body_order, minimum_pericenters, verbose_output = False):

    #____________________________transfer legs_______________________
    transfer_leg_settings, transfer_node_settings = (
        transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
            transfer_body_order,
            departure_orbit=(departure_semi_major_axis, departure_eccentricity),
            arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity),
            minimum_pericenters=minimum_pericenters,
        )
    )

    transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies,
        transfer_leg_settings,
        transfer_node_settings,
        transfer_body_order,
        central_body,
    )

    # Lower and upper bound on departure date
    departure_date_lb = DateTime(2000, 1, 1).to_epoch()
    departure_date_ub = DateTime(2001, 6, 20).to_epoch()

    # List of lower and upper on time of flight for each leg
    legs_tof_lb = np.zeros(7)
    legs_tof_ub = np.zeros(7)


    for i in range(len(legs_tof_lb)):
        legs_tof_lb[i] = 1 * constants.JULIAN_DAY
        legs_tof_ub[i] = 1 *  constants.JULIAN_YEAR
        
    optimizer = TransferTrajectoryProblem(
        transfer_trajectory_object,
        departure_date_lb,
        departure_date_ub,
        legs_tof_lb,
        legs_tof_ub,
    )

    # Creation of the pygmo problem object
    prob = pg.problem(optimizer)

    # Define number of generations per evolution
    number_of_generations = 3

    # Fix seed
    optimization_seed = 4444

    # Create pygmo algorithm object
    algo = pg.algorithm(pg.de(gen=number_of_generations, seed=optimization_seed, F=0.5))

    algo.set_verbosity(0)
    population_size = 120

    # Create population
    pop = pg.population(prob, size=population_size, seed=optimization_seed)

    ###########################################################################
    # Run optimization
    ###########################################################################

    # Set number of evolutions
    number_of_evolutions = 2000

    # Initialize empty containers
    individuals_list = []
    fitness_list = []

    for i in range(number_of_evolutions):

        pop = algo.evolve(pop)

        # individuals save
        individuals_list.append(pop.champion_x)
        fitness_list.append(pop.champion_f)
        #print(f"{i / number_of_evolutions * 100} %")
    au = 6.9e7
    node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(
    transfer_trajectory_object, pop.champion_x
    )   
    transfer_trajectory_object.evaluate(
        node_times, leg_free_parameters, node_free_parameters
    )
    state_history_dict = transfer_trajectory_object.states_along_trajectory(500)
    fly_by_states_raw = np.array([state_history_dict[node_times[i]] for i in range(len(node_times))])
    state_history = result2array(state_history_dict)
    times = state_history[:, 0]


    print("\n########### DELTA V PER NODE ###########")
    for i in range(len(transfer_body_order)):
        node_dv = transfer_trajectory_object.delta_v_per_node[i]
        print(f" - Departure/Arrival/Flyby at {transfer_body_order[i]}: {node_dv:.3f} m/s")
    print(f"Total => {pop.champion_f[0]} m/s")
    print("=======================================\n")

    plt.show()
    return pop.champion_f[0]
   # print("The optimization has finished")

minimum_pericenters = {
    "Callisto": 2634.1e3 + 100e3,
    "Ganymede": 2634.1e3 + 100e3, 
    "Europa":   1560.8e3 + 100e3,  
    "Io":       1821.6e3 + 100e3, 
    "target_orbit" : 0.1
}

transfers =  ["target_orbit", "Callisto", "Callisto", "Ganymede", "Ganymede"]
optimize(transfers, minimum_pericenters)

    # 4. Sort results to find the absolute best planetary sequence
    # results.sort(key=lambda x: x[0])
    # print("\n--- OPTIMIZATION RANKINGS ---")
    # for rank, (score, order) in enumerate(results, 1):
    #     print(f"Rank {rank}: Delta-V = {score:.2f} m/s | Sequence: {order}")
###########################################################################
# Results post-processing
###########################################################################

# Extract the best individual

# Reevaluate the transfer trajectory using the champion design variables


# # Manually test a single trajectory evaluation
# from tudatpy.astro.time_representation import DateTime


# au = 6.9e7

# # Extract the state history
# state_history_dict = transfer_trajectory_object.states_along_trajectory(500)
# fly_by_states_raw = np.array([state_history_dict[node_times[i]] for i in range(len(node_times))])
# print(fly_by_states_raw)
# state_history = result2array(state_history_dict)

# # Get Jupiter's position at each time to subtract
# from tudatpy.kernel.astro import element_conversion

# # Subtract Jupiter's state to get Jupiter-centered positions
# # state_history columns: [time, x, y, z, vx, vy, vz]
# times = state_history[:, 0]

# jupiter_states = np.array([
#     spice.get_body_cartesian_state_at_epoch("Jupiter", "SSB", "ECLIPJ2000", "NONE", t)
#     for t in times
# ])

# state_history_jc = state_history.copy()
# fly_by_states_jc = fly_by_states_raw.copy()


# fig = plt.figure(figsize=(8, 5))
# ax = fig.add_subplot(111)
# ax.plot(state_history_jc[:, 1] / au, state_history_jc[:, 2] / au)

# colors = {
#     "Callisto"  : "brown",
#     "Ganymede"  : "red",
#     "Europa"    : "green",
#     "Io"        : "blue",
#     "target_orbit"        : "yellow",
# }

# for i, body in enumerate(transfer_body_order):
#     ax.scatter(fly_by_states_jc[i, 0] / au, fly_by_states_jc[i, 1] / au,
#             color=colors[body], label=body, zorder=5)


# ax.scatter([0], [0], color="orange", label="Jupiter", s=200, zorder=5)

# ax.set_xlabel("x R_j")
# ax.set_ylabel("y R_j ")
# ax.set_aspect("equal")
# ax.legend(bbox_to_anchor=[1, 1])
# plt.tight_layout()




