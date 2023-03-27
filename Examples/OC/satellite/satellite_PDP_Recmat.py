from PDP import PDP
from MyEnv import MyEnv
import numpy as np
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import time

# --------------------------- load environment ----------------------------------------
satellite = MyEnv.Satellite()
satellite.initDyn(n=0.001038159246831192)
satellite.initCost(Q=np.block([[np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), 1000 * np.eye(3)]]),
                   R=1e10 * np.eye(3))

# --------------------------- create PDP Control/Planning object ----------------------------------------
dt = 0.1
horizon = 50
OCP = PDP.OCSys()
OCP.setStateVariable(satellite.X)
OCP.setControlVariable(satellite.U)
dyn = satellite.X + dt * satellite.f
OCP.setDyn(dyn)
OCP.setPathCost(satellite.path_cost)
OCP.setFinalCost(satellite.final_cost)
ini_X = np.array([0.00000000e+00,
                  1.00000000e+03,
                  0.00000000e+00,
                  5.19079623e-01,
                  -0.00000000e+00,
                  1.03815925e+00])

ini_state = ini_X

# --------------------------- create PDP true oc solver ----------------------------------------
true_OCP = PDP.OCSys()
true_OCP.setStateVariable(satellite.X)
true_OCP.setControlVariable(satellite.U)
true_OCP.setDyn(dyn)
true_OCP.setPathCost(satellite.path_cost)
true_OCP.setFinalCost(satellite.final_cost)
true_sol = true_OCP.ocSolver(ini_state=ini_state, horizon=horizon)
true_state_traj = true_sol['state_traj_opt']
true_control_traj = true_sol['control_traj_opt']
print(true_sol['cost'])
# satellite.play_animation(satellite_len=2,state_traj=true_sol['state_traj_opt'],control_traj=true_sol['control_traj_opt'])
# ---------------------------- start learning the control policy -------------------------------------
for j in range(5):  # tial loop
    start_time = time.time()
    # learning rate
    lr = 1e-4
    # initialize
    loss_trace, parameter_trace = [], []
    OCP.recmat_init_step(horizon, -1)
    # current_parameter = np.random.randn(OCP.n_auxvar)
    current_parameter = true_sol['control_traj_opt'].flatten() + 5 * np.random.randn(
        true_sol['control_traj_opt'].flatten().size)
    parameter_trace += [current_parameter.flatten()]
    # iteration
    for k in range(int(5e4)):  # maximum iteration for each trial
        # one iteration of PDP
        loss, dp = OCP.recmat_step(ini_state, horizon, current_parameter)
        # update
        current_parameter -= lr * dp
        loss_trace += [loss]
        parameter_trace += [current_parameter.flatten()]
        # print
        if k % 100 == 0:
            print('Trial:', j, 'Iter:', k, 'loss:', loss)

    # solve the trajectory
    sol = OCP.recmat_unwarp(ini_state, horizon, current_parameter)
    # satellite.play_animation(satellite_len=2,state_traj=sol['state_traj'],control_traj=sol['control_traj'])

    # save the results
    save_data = {'trail_no': j,
                 'parameter_trace': parameter_trace,
                 'loss_trace': loss_trace,
                 'learning_rate': lr,
                 'solved_solution': sol,
                 'true_solution': true_sol,
                 'time_passed': time.time() - start_time,
                 'dt': dt,
                 'horizon': horizon
                 }
    sio.savemat('./data/PDP_OC_results_trial_' + str(j) + '.mat', {'results': save_data})
