import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm





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
ground_truth_parameters = torch.tensor([2.0, 2.0, 3.0, 3.0])
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
# everything is in numpy
def construct_mpc_problem():
    x = cp.Parameter(n) # this is the parameters of the problem
    acceleration_of_other_car = cp.Parameter(1) # this is the parameters of the problem
    states = [cp.Variable(n) for _ in range(T)] # this is the decision variables of MPC
    controls = [cp.Variable(m) for _ in range(T)] # this is the decision variables of MPC
    # initial constraints
    constraints = [states[0] == x, cp.norm(controls[0], 'inf') <= control_limit] 
    # initial objective
    objective = cp.sum(cp.multiply(ground_truth_parameters_np, cp.square(states[0]))) +\
        cp.sum_squares(controls[0]) 
    for t in range(1, T):
        # objective
        objective += cp.sum(cp.multiply(ground_truth_parameters_np, cp.square(states[t]))) +\
            cp.sum_squares(controls[t]) 
        # dynamics constraints
        constraints += [states[t] == A @ states[t-1] +\
            B1 @ controls[t-1] +\
            B2 @ acceleration_of_other_car] 
        # control constraints
        constraints += [cp.norm(controls[t], 'inf') <= control_limit] 
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
        costs.append(cost(xt, ut).item())
        states.append(dynamics_torch(xt, ut, acceleration_of_other_car))
    return states[:-1], controls, costs


# define the mean square error
def mse(prediction, actual):
    prediction = torch.stack(prediction, dim=0)
    actual = torch.stack(actual, dim=0)
    return (prediction - actual).pow(2).mean(axis=1).mean(axis=0).item()





# ----------------- 2. Differentiable MPC -----------------



# define the differentiable MPC problem as below
# The problem bellow is the same as the MPC problem except that the parameters are inferred
def construct_differentiable_mpc_problem():
    x = cp.Parameter(n) # this is the parameters of the problem
    inferred_parameters = cp.Parameter(n, nonneg=True) # this is the parameters of the problem
    acceleration_of_other_car = cp.Parameter(1) # this is the parameters of the problem
    states = [cp.Variable(n) for _ in range(T)] # this is the decision variables of MPC
    controls = [cp.Variable(m) for _ in range(T)] # this is the decision variables of MPC
    # initial constraints
    constraints = [states[0] == x, cp.norm(controls[0], 'inf') <= control_limit] 
    # initial time state and control cost
    objective = cp.sum(cp.multiply(inferred_parameters, cp.square(states[0]))) + \
    cp.sum_squares(controls[0]) 
    for t in range(1, T):
        # instantaneous state and control cost
        objective += cp.sum(cp.multiply(inferred_parameters, cp.square(states[t]))) + \
        cp.sum_squares(controls[t]) 
        # dynamics constraints
        constraints += [states[t] == A @ states[t-1] +  
                        B1 @ controls[t-1] +  
                        B2 @ acceleration_of_other_car] 
        # control constraints
        constraints += [cp.norm(controls[t], 'inf') <= control_limit] 
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return CvxpyLayer(problem, variables=[controls[0]], parameters=[x, acceleration_of_other_car, inferred_parameters])





# ----------------- 3. test MPC and differentiable MPC -----------------



# initial state of two cars
x0 = torch.tensor([1., 1., 0., 0.])
acceleration_of_other_car = torch.tensor([0.])


# define an MPC problem:
# In this MPC problem, we want to solve for the control given the initial state x0 and 
# the acceleration of the other car
mpc_problem = construct_mpc_problem()


# for example, we can solve the MPC problem for the initial state x0 and the acceleration of the other car
# Notice that the output of the function construct_mpc_problem is the first control, i.e., controls[0]
mpc_problem(x0, acceleration_of_other_car)


# simulate the closed-loop system and get expert state and control trajectories, and costs
expert_states, expert_controls, expert_costs = simulate(mpc_problem, 
                                                        x0, 
                                                        acceleration_of_other_car, 
                                                        n_iters = 300)




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
inferred_parameters_torch = torch.ones(n, requires_grad=True)

# the training loop:
training_epochs = 40 
test_losses = [] 
training_losses = [] 
x0 = torch.tensor([1., 1., 0., 0.])
test_x0 = torch.tensor([1., 1., 0., 0.])
# initialize the training 
with torch.no_grad():
    _, initial_control_prediction, _ = simulate(inferred_policy, test_x0, acceleration_of_other_car, n_iters = 100)
    _, test_expert_control, _ = simulate(expert_policy, test_x0, acceleration_of_other_car, n_iters = 100)
    test_losses.append(mse(initial_control_prediction, test_expert_control))
    print(test_losses[-1])


# use Adam optimizer
opt = torch.optim.Adam([inferred_parameters_torch], lr=1e-3) 


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
        _, inferred_control_prediction, _ = simulate(inferred_policy, 
                                                    test_x0, 
                                                    acceleration_of_other_car, 
                                                    n_iters = 100)
        test_losses.append(mse(inferred_control_prediction, test_expert_control))
    print(test_losses[-1])


print('Training finished!')

print('The inferred parameters are: ', inferred_parameters_torch.data)
print('The ground truth parameters are: ', ground_truth_parameters.data)

