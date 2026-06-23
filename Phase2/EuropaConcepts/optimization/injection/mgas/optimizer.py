from injection.mgas.main import*

###########################################################################
# Setup optimization
###########################################################################
# Initialize optimization class


def optimize(transfer_body_order, minimum_pericenters, progress_dict, verbose_output = False):

    process_key = " ➔ ".join(transfer_body_order)
    progress_dict[process_key] = 0
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
    population_size = 200

    # Create population
    pop = pg.population(prob, size=population_size, seed=optimization_seed)

    ###########################################################################
    # Run optimization
    ###########################################################################

    # Set number of evolutions
    number_of_evolutions = 3000

    # Initialize empty containers
    individuals_list = []
    fitness_list = []

    for i in range(number_of_evolutions):

        pop = algo.evolve(pop)

        # individuals save
        individuals_list.append(pop.champion_x)
        fitness_list.append(pop.champion_f)
        progress_dict[process_key] = i + 1
        #print(f"{i / number_of_evolutions * 100} %")
    if verbose_output:
        clean_filename = "_".join(transfer_body_order) + ".png"
        print("\n########### CHAMPION INDIVIDUAL ###########\n")
        print("Total Delta V [m/s]: ", pop.champion_f[0])
        best_decision_variables = pop.champion_x / constants.JULIAN_DAY
        print(pop.champion_x)
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
        #ax.set_ylim([4, 25])
        ax.grid("major")
        ax.set_title("Best individual over generations", fontweight="bold")
        ax.set_xlabel("Number of generation")
        ax.set_ylabel(r"$\Delta V [km/s]$")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.legend()
        plt.savefig(clean_filename, dpi=300)
        plt.close()
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

    jupiter_states = np.array([
        spice.get_body_cartesian_state_at_epoch("Jupiter", "SSB", "ECLIPJ2000", "NONE", t)
        for t in times
    ])

    state_history_jc = state_history.copy()
    fly_by_states_jc = fly_by_states_raw.copy()


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(state_history_jc[:, 1] / au, state_history_jc[:, 2] / au)

    colors = {
        "Callisto"  : "brown",
        "Ganymede"  : "red",
        "Europa"    : "green",
        "Io"        : "blue",
        "target_orbit"        : "yellow",
    }

    for i, body in enumerate(transfer_body_order):
        ax.scatter(fly_by_states_jc[i, 0] / au, fly_by_states_jc[i, 1] / au,
                color=colors[body], label=body, zorder=5)


    ax.scatter([0], [0], color="orange", label="Jupiter", s=200, zorder=5)

    ax.set_xlabel("x R_j")
    ax.set_ylabel("y R_j ")
    ax.set_aspect("equal")
    ax.legend(bbox_to_anchor=[1, 1])
    plt.title(f"Delta V {pop.champion_f[0]} m/s =>_".join(transfer_body_order[1:]))
    #plt.tight_layout()
    plt.legend()
    clean_filename = "_".join(transfer_body_order[1:]) + "_TRAJECTORIES.png"
    plt.savefig(clean_filename, dpi=300)
    plt.close()
    return pop.champion_f[0]
   # print("The optimization has finished")


#best_value, transfer_trajectory_object, node_times = optimize(transfer_body_order=transfer_body_order, minimum_pericenters=minimum_pericenters)

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import time 

if __name__ == "__main__":
    # Force 'spawn' method for clean C++ memory allocation
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    list_of_orders = [
        #  ["target_orbit", "Ganymede", "Ganymede"],
        #  ["target_orbit", "Callisto", "Ganymede", "Ganymede"],
        #  ["target_orbit", "Callisto", "Ganymede", "Ganymede", "Ganymede"],
        #  ["target_orbit", "Ganymede", "Ganymede", "Ganymede", "Ganymede"],
        #  ["target_orbit", "Callisto", "Callisto", "Ganymede", "Ganymede"],
        #  ["target_orbit", "Callisto", "Ganymede", "Callisto", "Ganymede"],
         ["target_orbit", "Callisto", "Callisto", "Callisto", "Ganymede", "Ganymede"],
         ["target_orbit", "Callisto", "Ganymede", "Callisto", "Ganymede", "Ganymede"],
         ["target_orbit", "Callisto", "Ganymede", "Callisto", "Ganymede", "Ganymede", "Ganymede"],
        #  ["target_orbit", "Ganymede", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Ganymede", "Ganymede", "Ganymede", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Ganymede", "Europa"],
        #  ["target_orbit", "Ganymede", "Callisto", "Ganymede", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Ganymede", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Europa", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Callisto", "Ganymede", "Ganymede", "Europa"],
        #  ["target_orbit", "Ganymede", "Ganymede", "Ganymede", "Ganymede", "Europa"],
        #  ["target_orbit", "Ganymede", "Ganymede", "Ganymede", "Ganymede", "Europa", "Europa", "Europa"],
        #  ["target_orbit", "Ganymede", "Ganymede", "Ganymede", "Ganymede", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Callisto", "Ganymede", "Ganymede", "Europa", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Ganymede", "Callisto", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Callisto", "Ganymede", "Europa", "Europa"],
        #  ["target_orbit", "Callisto", "Callisto", "Ganymede", "Ganymede", "Europa"],
        #  ["target_orbit", "Callisto", "Ganymede", "Callisto", "Ganymede", "Europa"],
    ]
    
    minimum_pericenters = {
        "Callisto": 2634.1e3 + 100e3,
        "Ganymede": 2634.1e3 + 100e3, 
        "Europa":   1560.8e3 + 100e3,  
        "Io":       1821.6e3 + 100e3, 
        "target_orbit" : 0.1
    }

    # 3. Create the shared dictionary managed by the main process
    manager = multiprocessing.Manager()
    shared_progress = manager.dict()
    
    # Initialize keys so the main process knows what to expect
    for order in list_of_orders:
        shared_progress[" ➔ ".join(order)] = 0

    num_cores = min(os.cpu_count(), len(list_of_orders))
    total_evos = 2500

    print(f"Launching optimizations across {num_cores} processes...\n")

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit tasks, passing the shared dictionary in
        futures = {
            executor.submit(optimize, order, minimum_pericenters, shared_progress, True): order 
            for order in list_of_orders
        }
        
        # 4. The Main Process Print Loop
        # While any background process is still running, print the iterations
        while not all(f.done() for f in futures):
            
            # Optional: Clear the terminal screen to make a static dashboard
            # On Linux/macOS use 'clear', on Windows use 'cls'
            os.system('clear' if os.name == 'posix' else 'cls') 
            
            print("================ CURRENT PROGRESS ================")
            for sequence, current_iteration in shared_progress.items():
                print(f"Sequence: {sequence:<45} | Iteration: {current_iteration}/{total_evos}")
            print("==================================================")
            
            time.sleep(1) # Wait 1 second before gathering the next update

        # 5. Collect final results once everything is finished
        print("\nAll optimizations complete!")
        print("\n=================== FINAL RESULTS ===================")
        for future in futures:
            order = futures[future]
            try:
                print(f"Order {order} -> Best Delta-V: {future.result():.2f} m/s")
            except Exception as e:
                print(f"Order {order} failed with error: {e}")

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


# print("\n########### DELTA V PER NODE ###########")
# for i in range(len(transfer_body_order)):
#     node_dv = transfer_trajectory_object.delta_v_per_node[i]
#     print(f" - Departure/Arrival/Flyby at {transfer_body_order[i]}: {node_dv:.3f} m/s")
# print(f"Total => {best_value} m/s")
# print("=======================================\n")

# plt.show()

