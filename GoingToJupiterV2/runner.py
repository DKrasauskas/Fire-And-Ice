
from main_calculator import run_sim
import numpy as np

#######################
# Start Values
#######################

transfer_body_order = ["Earth","Mars", "Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,500])
legs_tof_ub = np.array([600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth", "Mars", "Earth", "Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([400,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Mars", "Earth","Venus", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Mars","Earth", "Venus","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,20,20,500])
legs_tof_ub = np.array([300,500,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Mars", "Mars","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,800,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Earth", "Jupiter"]
legs_tof_lb = np.array([20,100,500])
legs_tof_ub = np.array([300,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Earth","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Earth","Mars", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Earth","Earth","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,300,500])
legs_tof_ub = np.array([300,600,800,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Mars","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Venus","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,500])
legs_tof_ub = np.array([300,600,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

transfer_body_order = ["Earth","Venus", "Venus","Earth","Earth", "Jupiter"]
legs_tof_lb = np.array([20,20,300,300,500])
legs_tof_ub = np.array([300,600,800,1200,1500])

run_sim(transfer_body_order, legs_tof_lb, legs_tof_ub)

