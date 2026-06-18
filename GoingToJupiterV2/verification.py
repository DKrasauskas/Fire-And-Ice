from main_calculator import run_sim
import numpy as np

####
#This file runs the different values for to compare the effact of optimisation population size, and different seeds.
####

transfer_body_order = ["Earth","Venus", "Earth","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=5)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=10)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=30)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=100)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=300)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=1000)
run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, pop_size=3000)

# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1000)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1001)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1002)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1003)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1004)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1005)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1006)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1007)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1008)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1009)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1010)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1011)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1012)
# run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub, seed=1013)
