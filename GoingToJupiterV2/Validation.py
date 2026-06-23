from validation.main_calc_validation import run_sim
import numpy as np

#######################
#Comparison to Galileo
#######################



transfer_body_order = ["Earth","Venus", "Earth", "Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

run_sim(transfer_body_order,legs_tof_lb,legs_tof_ub)