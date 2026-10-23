from itertools import product
import time as t

import numpy as np
from matplotlib import pyplot as plt

from nbody import euler_step, euler_cromer_step, velocity_verlet_step, hermite_step, iterated_hermite_step, heun_step, rk4_step, G, D

def load_test_data(path):
    data = np.loadtxt(path, skiprows=3).reshape((-1, 2, 13)).transpose((0, 2, 1))
    positions = data[:, 2:5, :]
    velocities = data[:, 5:8, :]
    masses = data[:, 8, :]
    energy = data[:, 9, :]
    specific_angular_momentum = data[:, 10, :]
    runge_lenz_vector = data[:, 11, :]
    semimajor_axis = data[:, 12, :]
    return (masses, positions, velocities, energy, specific_angular_momentum, runge_lenz_vector, semimajor_axis)


def load_initial_data(path):
    data = np.loadtxt(path).reshape((-1, 7)).transpose()
    N = data.shape[1]

    masses = data[6, :]
    masses /= np.sum(masses)

    positions = data[0:3, :]
    center_of_mass = (positions @ masses).repeat(N).reshape((3, N))
    positions -= center_of_mass

    velocities = data[3:6, :]
    center_of_velocities = (velocities @ masses).repeat(N).reshape((3, N))
    velocities -= center_of_velocities


    return (N, masses, positions, velocities)


## ==== TASK 2 ====

def calculate_orbit(N, masses, positions, velocities, integrator, dt, steps):
    position_history = []
    velocity_history = []

    for step in range(steps):
        (positions, velocities) = integrator(N, masses, positions, velocities, dt)
        position_history.append(positions)
        velocity_history.append(velocities)

    return (np.array(position_history), np.array(velocity_history))


(N, masses, initial_positions, initial_velocities) = load_initial_data('in2')

def conserved_2b_quantities(N, masses, positions, velocities):
    M = np.sum(masses)
    r = positions[:, :, 1] - positions[:, :, 0]
    v = velocities[:, :, 1] - velocities[:, :, 0]

    # Energy
    pos_matrix = np.triu(1/np.linalg.norm(positions[:, :, :, None] - positions[:, :, None, :], axis=1), k=1)
    E = 0.5 * (np.linalg.norm(velocities, axis=1)**2) @ masses - G * np.einsum('i,kij,j->k', masses, pos_matrix, masses)
    # Specific angular momentum j
    j = np.cross(r, v, axis=1)

    # Runge-lenz vector e, eccentricity |e|
    e = np.cross(v, j, axis=1)/(G * M) - r/np.linalg.norm(r, axis=1, keepdims=True)

    # Semimajor axis a
    a = (np.linalg.norm(j, axis=1)**2 / (G * M)) / (1-np.linalg.norm(e, axis=1)**2)
    return (E, j, e, a)


# ==== TEST CORRECTNESS OF IMPLEMENTATION ====


def assert_small(array):
    assert not np.any(np.abs(array) > 1e-8)

def test_integrator(testfile, integrator):
    steps = 100
    dt = 0.01

    (cal_positions, cal_velocities) = calculate_orbit(N, masses, initial_positions, initial_velocities, integrator, dt, steps)
    (E, j, e, a) = conserved_2b_quantities(N, masses, cal_positions, cal_velocities)

    (_, true_positions, true_velocities, true_energy, true_j, true_e, true_a) = load_test_data(testfile)

    j_diff = np.linalg.norm(j, axis=1)[0:steps] - true_j[0:steps, 1]
    e_diff = np.linalg.norm(e, axis=1)[0:steps] - true_e[0:steps, 1]
    a_diff = a[0:steps] - true_a[0:steps, 1]
    E_diff = E[0:steps] - true_energy[0:steps, 1]
    position_diff = cal_positions[0:steps, :, :] - true_positions[0:steps, :, :]
    
    assert_small(position_diff)
    assert_small(j_diff)
    assert_small(e_diff)
    assert_small(a_diff)
    assert_small(E_diff)

test_integrator('euler0.01', euler_step)
test_integrator('euler_cromer0.01', euler_cromer_step)
test_integrator('hermite0.01', hermite_step)
test_integrator('iterated_hermite0.01', iterated_hermite_step)
test_integrator('rk40.01', rk4_step)
print("ALL TESTS PASSED!")



# ==== GENERATE SIMULATIONS FOR THE PROGOCOL ====

nus = [0.5, 0.1, 0.05, 0.01]
integrators = [euler_step, euler_cromer_step, velocity_verlet_step, hermite_step, iterated_hermite_step, heun_step, rk4_step]
t_max = 2

(N, masses, initial_positions, initial_velocities) = load_initial_data('in2')

for (dt, integrator) in product(nus, integrators):
    steps = int(t_max / dt)
    time = np.array([(i+1)*dt for i in range(steps)])
    (cal_positions, cal_velocities) = calculate_orbit(N, masses, initial_positions, initial_velocities, integrator, dt, steps)
    (E, j, e, a) = conserved_2b_quantities(N, masses, cal_positions, cal_velocities)
    j = np.linalg.norm(j, axis=1)
    e = np.linalg.norm(e, axis=1)

    plt.title(f"2 Body Problem, nu={dt}, {integrator.__name__.replace("_"," ").replace(" step", "")}")
    plt.xlabel("time / s")
    plt.ylabel("relative log error")
    plt.plot(time, np.log(np.abs((E[0] - E)/E[0])), label='energy')
    plt.plot(time, np.log(np.abs((j[0] - j)/j[0])), label='specific angular momentum')
    plt.plot(time, np.log(np.abs((e[0] - e)/e[0])), label='runge lenz vector')
    plt.plot(time, np.log(np.abs((a[0] - a)/a[0])), label='semimajor axis')

    plt.legend()

    plt.savefig(f"2body_{dt}_{integrator.__name__}.png")
    plt.close()


# varying elipticity

(N, masses, initial_positions, initial_velocities) = load_initial_data('in2')

dt = 0.01
steps = int(t_max / dt)

dvys = [0.01 * (i-50) for i in range(100)]
es = []
E_errors = []
j_errors = []
e_errors = []
a_errors = []
for dvy in dvys:
    initial_positions[1, 1] = dvy - 1
    (cal_positions, cal_velocities) = calculate_orbit(N, masses, initial_positions, initial_velocities, euler_step, dt, steps)
    (E, j, e, a) = conserved_2b_quantities(N, masses, cal_positions, cal_velocities)
    j = np.linalg.norm(j, axis=1)
    e = np.linalg.norm(e, axis=1)
    es.append(e[0])
    E_errors.append(np.log(np.abs((E[0] - E[steps-1])/E[0])))
    j_errors.append(np.log(np.abs((j[0] - j[steps-1])/j[0])))
    e_errors.append(np.log(np.abs((e[0] - e[steps-1])/e[0])))
    a_errors.append(np.log(np.abs((a[0] - a[steps-1])/a[0])))

es = np.array(es)
E_errors = np.array(E_errors)
j_errors = np.array(j_errors)
e_errors = np.array(e_errors)
a_errors = np.array(a_errors)

plt.close()
plt.title("Relative error vs eccentricity, euler")
plt.xlabel("eccentricity")
plt.ylabel(f"relative log error at t={t_max}s")
plt.plot(es, E_errors, label='energy')
plt.plot(es, j_errors, label='specific angular momentum')
plt.plot(es, e_errors, label='eccentricity')
plt.plot(es, a_errors, label='semimajor axis')
plt.legend()
plt.savefig("2body_rel_error_vs_eccentricity.png")
plt.close()

# ==== EXERCISE 3 ====


def accuracy_vs_nu(path, name):
    (N, masses, initial_positions, initial_velocities) = load_initial_data(path)

    dt_steps = 100
    dts = [(0.1 * (dt_steps-i) + 0.01 * i)/dt_steps for i in range(dt_steps)]
    errors = []
    times = []

    start = t.time() 
 
    for dt in dts:
        (cal_positions, cal_velocities) = calculate_orbit(N, masses, initial_positions, initial_velocities, euler_step, dt, steps)
        (E, _, _, _) = conserved_2b_quantities(N, masses, cal_positions, cal_velocities)
        times.append(t.time()-start)
        errors.append(np.log(np.abs((E[0] - E[steps-1])/E[0])))

    errors = np.array(errors)
    times = np.array(times)

    plt.title("nu vs errors")
    plt.xlabel("nu (dt)")
    plt.ylabel("log relative energy error")
    plt.plot(dts, errors)
    plt.savefig(f"nu_vs_errors_{name}")
    plt.close()

    plt.title("nu vs time")
    plt.ylabel(f"time for {steps} steps in seconds")
    plt.xlabel("nu (dt)")
    plt.plot(dts, times)
    plt.savefig(f"nu_vs_times_{name}")
    plt.close()

accuracy_vs_nu('in100', '100_particles')

