import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm



# lessons learned:
# 1. the problem should feasible, convex, for any given parameters
# 2. Make sure the there is no product of two parameters that to be inferred



# ----------------- 1. Define the MPC problem -----------------



from cvxpylayers.torch import CvxpyLayer
torch.set_default_dtype(torch.double)
torch.set_default_device('cuda:3')

# define the nominal dynamics
n = 4 # state dimension
m = 1 # control dimension
dt = 0.1 # time integrator
A_np = np.array([[1., dt],[0, 1.]]) # dynamics of the ego car
A = np.kron(np.eye(2), A_np) # dynamics of the two cars
B1 = np.array([0,dt,0,0]).reshape(-1,1) # control of the ego car
B2 = np.array([0,0,0,dt]).reshape(-1,1) # control of the other car

A_torch = torch.tensor(A)
B1_torch = torch.tensor(B1)
B2_torch = torch.tensor(B2)


# define the ground truth parameters of the cost function
# parameters:   weight_on_distance_cost, 
#               weight_on_velocity_cost, 
#               weight_on_acceleration_cost, 
#               desired_distance, 
#               desired_velocity,
ground_truth_parameters = torch.tensor([4.0, 1.0, 0.5, 1.0])
ground_truth_parameters_np = ground_truth_parameters.cpu().numpy()
control_limit = 1.0 # this is the constraint on the control
T = 5 # look-ahead horizon


# define the dynamics and cost function
def dynamics_torch(xt, ut, acceleration_of_other_car = torch.tensor([0.])):
    return A_torch @ xt + B1_torch @ ut + B2_torch @ acceleration_of_other_car

def dynamics_np(xt, ut, acceleration_of_other_car = np.array([0.])):
    return A @ xt + B1 @ ut + B2 @ acceleration_of_other_car

def cost(xt, ut):
    return (ground_truth_parameters*(xt.pow(2))).sum() + ut.pow(2).sum()


# define MPC problem as below:
def construct_mpc_problem():
    # belows are the parameters of the MPC problem
    x = cp.Parameter(n)
    acceleration_of_other_car = cp.Parameter(1)
    # belows are the decision variables of the MPC problem
    states = [cp.Variable(n) for _ in range(T)]
    controls = [cp.Variable(m) for _ in range(T)]
    # initial constraints
    constraints = [states[0] == x, 
        cp.norm(controls[0], 'inf') <= control_limit,
        states[0][0] <= states[0][2] - ground_truth_parameters_np[2]] 
    # initial objective
    # 
    objective = cp.square(ground_truth_parameters_np[2]-(states[0][2]-states[0][0])) +\
        cp.multiply(ground_truth_parameters_np[0], cp.square(states[0][1] - ground_truth_parameters_np[3]))+\
        cp.multiply(ground_truth_parameters_np[1], cp.square(controls[0])) 
    for t in range(1, T):
        # objective
        #
        objective = cp.square(ground_truth_parameters_np[2]-(states[t][2]-states[t][0])) +\
            cp.multiply(ground_truth_parameters_np[0], cp.square(states[t][1] - ground_truth_parameters_np[3]))+\
            cp.multiply(ground_truth_parameters_np[1], cp.square(controls[t])) 
        # dynamics constraints
        constraints += [states[t] == A @ states[t-1] +\
            B1 @ controls[t-1] +\
            B2 @ acceleration_of_other_car] 
        # control constraints
        constraints += [cp.norm(controls[t], 'inf') <= control_limit]
        constraints += [states[t][0] <= states[t][2]- ground_truth_parameters_np[2]] 
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, variables=[controls[0]], parameters=[x, acceleration_of_other_car])


# simulate the closed-loop system
def simulate(policy, x0, acceleration_of_other_car, n_iters=1):
    states, controls, costs = [x0], [], []
    for t in tqdm(range(n_iters)):
        xt = states[-1] # current state
        # import pdb; pdb.set_trace()
        ut = policy(xt, acceleration_of_other_car)[0] # current control
        # record control, cost, and next state:
        controls.append(ut)
        states.append(dynamics_torch(xt, ut, acceleration_of_other_car))
    return states[:-1], controls


# define the mean square error
def mse(prediction, actual):
    prediction = torch.stack(prediction, dim=0)
    actual = torch.stack(actual, dim=0)
    return (prediction - actual).pow(2).mean(axis=1).mean(axis=0).item()





# ----------------- 2. Differentiable MPC -----------------



# define the differentiable MPC problem as below
# The problem bellow is the same as the MPC problem except that the parameters are inferred
def construct_differentiable_mpc_problem():
    # belows are the parameters of the MPC problem
    x = cp.Parameter(n)
    inferred_parameters = cp.Parameter(4, nonneg=True) # this is the parameters of the problem
    acceleration_of_other_car = cp.Parameter(1)
    # belows are the decision variables of the MPC problem
    states = [cp.Variable(n) for _ in range(T)]
    controls = [cp.Variable(m) for _ in range(T)]
    # initial constraints
    constraints = [states[0] == x, 
        cp.norm(controls[0], 'inf') <= control_limit,
        states[0][0] <= states[0][2] - inferred_parameters[2]] 
    # initial objective
    # 
    objective = cp.square(inferred_parameters[2]-(states[0][2]-states[0][0])) +\
        cp.multiply(inferred_parameters[0], cp.square(states[0][1] - ground_truth_parameters_np[3]))+\
        cp.multiply(inferred_parameters[1], cp.square(controls[0])) 
    for t in range(1, T):
        # objective
        #
        objective = cp.square(inferred_parameters[2]-(states[t][2]-states[t][0])) +\
            cp.multiply(inferred_parameters[0], cp.square(states[t][1] - ground_truth_parameters_np[3]))+\
            cp.multiply(inferred_parameters[1], cp.square(controls[t])) 
        # dynamics constraints
        constraints += [states[t] == A @ states[t-1] +\
            B1 @ controls[t-1] +\
            B2 @ acceleration_of_other_car] 
        # control constraints
        constraints += [cp.norm(controls[t], 'inf') <= control_limit]
        constraints += [states[t][0] <= states[t][2]- inferred_parameters[2]] 
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, variables=[controls[0]], parameters=[x, acceleration_of_other_car, inferred_parameters])




# ----------------- 3. test MPC and differentiable MPC -----------------



# initial state of two cars
x0 = torch.tensor([1.0, 0.2, 5.0, 1.0])
acceleration_of_other_car = torch.tensor([0.])


# define an MPC problem:
# In this MPC problem, we want to solve for the control given the initial state x0 and 
# the acceleration of the other car
mpc_problem = construct_mpc_problem()


# for example, we can solve the MPC problem for the initial state x0 and the acceleration of the other car
# Notice that the output of the function construct_mpc_problem is the first control, i.e., controls[0]
mpc_problem(x0, acceleration_of_other_car)




# generate expert state and control trajectories!
expert_states, expert_controls= simulate(mpc_problem, 
                                        x0, 
                                        acceleration_of_other_car, 
                                        n_iters = 100)





# define an MPC problem containing some hidden parameters that we want to infer
inferred_mpc_problem = construct_differentiable_mpc_problem()


# define the inferred policy: a function from a state to a control, under the inferred parameters
inferred_policy = lambda x, acceleration_of_other_car: inferred_mpc_problem(x, 
                                                acceleration_of_other_car, 
                                                inferred_parameters_torch)


# define the expert policy: a function from a state to a control, under the ground truth parameters
expert_policy = lambda x, acceleration_of_other_car: mpc_problem(x, 
                                    acceleration_of_other_car)





# ----------------- 4. training -----------------


# define the inferred parameters:
inferred_parameters_torch = torch.tensor([4.0, 1.0, 0.5, 1.0], requires_grad=True)

# the training loop:
training_epochs = 40 
test_losses = [] 
training_losses = [] 
test_x0 = x0
# initialize the training 
with torch.no_grad():
    _, initial_control_prediction = simulate(inferred_policy, test_x0, acceleration_of_other_car, n_iters = 100)
    _, test_expert_control = simulate(expert_policy, test_x0, acceleration_of_other_car, n_iters = 100)
    test_losses.append(mse(initial_control_prediction, test_expert_control))
    print(test_losses[-1])


# use Adam optimizer
opt = torch.optim.Adam([inferred_parameters_torch], lr=1e-4) 


# training begins!
for epoch in range(training_epochs):
    print('Epoch: ', epoch)
    for xt, ut in tqdm(zip(expert_states, expert_controls)):
        opt.zero_grad()
        ut_hat = inferred_mpc_problem(xt, 
                                    acceleration_of_other_car, 
                                    inferred_parameters_torch)[0] # the output of the function is (controls[0])
        loss = (ut - ut_hat).pow(2).mean()
        loss.backward()
        training_losses.append(loss.item())
        opt.step()
    with torch.no_grad():
        inferred_parameters_torch.data = inferred_parameters_torch.relu()
        _, inferred_control_prediction = simulate(inferred_policy, 
                                                    test_x0, 
                                                    acceleration_of_other_car, 
                                                    n_iters = 100)
        test_losses.append(mse(inferred_control_prediction, test_expert_control))
    print(test_losses[-1])


print('Training finished!')

print('The inferred parameters are: ', inferred_parameters_torch.data)
print('The ground truth parameters are: ', ground_truth_parameters.data)






predicted_trajectory, predicted_control = [], []
expert_trajectory, expert_control = [], []
with torch.no_grad():
    predicted_trajectory, predicted_control = simulate(inferred_policy, test_x0, acceleration_of_other_car, n_iters = 100)
    expert_trajectory, expert_control = simulate(expert_policy, test_x0, acceleration_of_other_car, n_iters = 100)
    test_losses.append(mse(initial_control_prediction, test_expert_control))
    print(test_losses[-1])



import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
plt.plot([x[0].cpu() for x in predicted_trajectory], [x[1].cpu() for x in predicted_trajectory], label='inferred trajectory')
plt.plot([x[0].cpu() for x in expert_trajectory], [x[1].cpu() for x in expert_trajectory], label='expert trajectory')
plt.legend()


