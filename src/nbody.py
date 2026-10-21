import numpy as np
from matplotlib import pyplot as plt

G = 1
M = 1
D = 3

def load_data(path):
    f = open(path)
    lines = f.read().split('\n')
    lines = [words  for line in lines if (words := [l for l in line.split(' ') if not l == ''])]

    first_line = lines.pop(0)
    (N, t_max, eta) = (int(first_line[0]), float(first_line[1]), float(first_line[2]))
    
    masses = []
    positions = []
    velocities = []

    for _ in range(N):
        masses.append(float(lines.pop(0)[0]))

    for _ in range(N):
        line = lines.pop(0)
        positions.append([float(line[0]), float(line[1]), float(line[2])])

    for _ in range(N):
        line = lines.pop(0)
        velocities.append([float(line[0]), float(line[1]), float(line[2])])


    return ((N, t_max, eta), np.array(masses), np.array(positions).T, np.array(velocities).T)
    
def difference_matrix(N, matrix):
    m = matrix.repeat(N).reshape((D, N, N))
    return m - m.transpose([0, 2, 1])

def corrected_diff_norm_matrix(N, diff_matrix):
    norms = np.linalg.norm(diff_matrix, axis=0).repeat(D).reshape((N, N, D)).transpose((2, 0, 1))
    return norms + np.eye(N).repeat(D).reshape((N, N, D)).transpose((2, 0, 1))


def calculate_accelarations(N, masses, positions):
    a = difference_matrix(N, positions)

    corrected_dividor = np.pow(corrected_diff_norm_matrix(N, a), 3)

    accelarations = a / corrected_dividor 
    accelarations = accelarations @ masses

    return -G * accelarations

def calculate_accelaration_grads(N, masses, positions, velocities):
    r = difference_matrix(N, positions)
    v = difference_matrix(N, velocities)

    corrected_norms = corrected_diff_norm_matrix(N, r)

    grads = ((v/np.pow(corrected_norms, 3)) - 3*(np.sum(v * r, axis=0).repeat(D).reshape(N, N, D).transpose([2, 0, 1]) / np.pow(corrected_norms, 5))*r)
    grads = -G * grads @ masses
    return grads


def euler_step(N, masses, positions, velocities, dt):
    accelarations = calculate_accelarations(N, masses, positions)
    new_velocities = velocities + dt * accelarations
    new_positions = positions + dt * velocities
    return (new_positions, new_velocities)

def euler_cromer_step(N, masses, positions, velocities, dt):
    accelarations = calculate_accelarations(N, masses, positions)
    new_velocities = velocities + dt * accelarations
    new_positions = positions + dt * new_velocities
    return (new_positions, new_velocities)

def velocity_verlet_step(N, masses, positions, velocities, dt):
    accelarations = calculate_accelarations(N, masses, positions)
    new_positions = positions + dt * velocities + 0.5 * dt**2 * accelarations
    new_accelarations = calculate_accelarations(N, masses, new_positions)
    new_velocities = velocities + 0.5 * (accelarations + new_accelarations) * dt
    return (new_positions, new_velocities)

def hermite_step(N, masses, positions, velocities, dt):
    accelarations = calculate_accelarations(N, masses, positions)
    grads = calculate_accelaration_grads(N, masses, positions, velocities)

    predicted_positions = positions + velocities * dt + 0.5 * dt**2 * accelarations + (1/6)*dt**3 * grads
    predicted_velocities = velocities + accelarations * dt + 0.5 * dt**2 * grads

    predicted_accelarations = calculate_accelarations(N, masses, predicted_positions)
    predicted_grads = calculate_accelaration_grads(N, masses, predicted_positions, predicted_velocities)

    acc2 = -6*(accelarations - predicted_accelarations)/(dt**2) - 2*(2*grads + predicted_grads)/dt
    acc3 = 12*(accelarations - predicted_accelarations)/(dt**3) + 6*(grads + predicted_grads)/(dt**2)

    corrected_positions = predicted_positions + (1/24) * acc2 * dt**4 + (1/120) * acc3 * dt**5
    corrected_velocities = predicted_velocities + (1/6) * acc2 * dt**3 + (1/24) * acc3 * dt**4

    return (corrected_positions, corrected_velocities)

def iterated_hermite_step(N, masses, positions, velocities, dt):
    accelarations = calculate_accelarations(N, masses, positions)
    grads = calculate_accelaration_grads(N, masses, positions, velocities)

    predicted_positions = positions + velocities * dt + 0.5 * dt**2 * accelarations + (1/6)*dt**3 * grads
    predicted_velocities = velocities + accelarations * dt + 0.5 * dt**2 * grads

    def correction(predicted_positions, predicted_velocities):
        predicted_accelarations = calculate_accelarations(N, masses, predicted_positions)
        predicted_grads = calculate_accelaration_grads(N, masses, predicted_positions, predicted_velocities)

        corrected_velocities = velocities + 0.5 * (predicted_accelarations + accelarations) * dt + (1/12) * (predicted_grads - grads) * dt**2
        corrected_positions = positions + 0.5 * (corrected_velocities + velocities) * dt + (1/12) * (predicted_accelarations - accelarations) * dt**2
        return (corrected_positions, corrected_velocities)

    (p1, v1) = correction(predicted_positions, predicted_velocities)
    (p2, v2) = correction(p1, v1)

    return (p2, v2)


def heun_step(N, masses, positions, velocities, dt):
    acc = calculate_accelarations(N, masses, positions)
    kv1 = dt * acc
    kr1 = dt * velocities
    kv2 = dt * calculate_accelarations(N, masses, positions + kr1)
    kr2 = dt * (velocities + kv1)
    new_positions = positions + 0.5*(kr1+kr2)
    new_velocities = velocities + 0.5*(kv1 + kv2)
    return (new_positions, new_velocities)

def rk4_step(N, masses, positions, velocities, dt):
    acc = calculate_accelarations(N, masses, positions)
    v1 = dt * acc
    r1 = velocities * dt
    v2 = calculate_accelarations(N, masses, positions + 0.5*r1) * dt
    r2 = (velocities + 0.5*v1) * dt 
    v3 = calculate_accelarations(N, masses, positions + 0.5*r2) * dt
    r3 = (velocities + 0.5*v2) * dt
    v4 = calculate_accelarations(N, masses, positions + r3) * dt
    r4 = (velocities + v3) * dt

    new_positions = positions + (1/6)*r1 + (1/3)*r2 + (1/3)*r3 + (1/6)*r4
    new_velocities = velocities + (1/6)*v1 + (1/3)*v2 + (1/3)*v3 + (1/6)*v4

    return (new_positions, new_velocities)


# ===== Physical Properties =====

def potential_energy(N, masses, positions):
    norms = np.linalg.norm(difference_matrix(N, positions), axis=0)
    norm_factors = ((1/ (norms + np.eye(N))) * (1-np.eye(N)))
    energy = G * (masses.T @ norm_factors @ masses ) / 2
    return energy

def kinetic_energy(masses, velocities):
    norms = np.linalg.norm(velocities, axis=0)
    return - 0.5 * masses @ norms**2 

def total_energy(N, masses, positions, velocities):
    return potential_energy(N, masses, positions) + kinetic_energy(masses, velocities)

def total_momentum(masses, velocities):
    return velocities @ masses

def total_angular_momentum(N, masses, positions, velocities):
    return np.cross(positions, velocities, axis=0) @ masses

# ===== 2 BODY SYSTEM =====

def specific_angular_momentum(masses, positions, velocities):
    r = positions[:, 1] - positions[:, 0]
    v = velocities[:, 1] - velocities[:, 0]
    return np.cross(r, v)

def runge_lenz_vector(masses, positions, velocities):
    vec_r = positions[:, 1] - positions[:, 0]
    vec_v = velocities[:, 1] - velocities[:, 0]
    vec_j = np.cross(vec_r, vec_v)

    return np.cross(vec_v, vec_j)/(G * np.sum(masses)) - vec_r/np.linalg.norm(vec_r)

def semimajor_axis(masses, positions, velocities):
    vec_r = positions[:, 1] - positions[:, 0]
    vec_v = velocities[:, 1] - velocities[:, 0]
    vec_j = np.cross(vec_r, vec_v)

    e = np.linalg.norm(np.cross(vec_v, vec_j)/(G * np.sum(masses)) - vec_r/np.linalg.norm(vec_r))
    j = np.linalg.norm(vec_j)
    return (j**2 / (G * np.sum(masses))) / (1-e**2)

# ===== TESTS =====

assert np.array_equal(
    calculate_accelarations(2, np.array([2, 1]), 
        np.array(
        [[1, 0],
         [0, 0],
         [0, 0]]
        )
    ),
    np.array(
        [[-1, 2],
         [0,  0],
         [0,  0]]
    )
)



# ===== EXERCISES =====


((N, t_max, dt), masses, positions, velocities) = load_data('v1/in2')
acc = calculate_accelarations(N, masses, positions)

print(positions)
print(velocities)


functions = [euler_step, euler_cromer_step, velocity_verlet_step, hermite_step, iterated_hermite_step, heun_step, rk4_step]

def two_body_test():
    ((N, t_max, dt), masses, positions, velocities) = load_data('v1/in2')

    for step_function in functions:
        ((N, t_max, dt), masses, positions, velocities) = load_data('v1/in2')

        time = []
        energy = []
        momentum = []
        angular_momentum = []
        j = []
        e = []
        a = []
        

        STEPS = int(t_max/dt)
        for i in range(STEPS + 1):

           energy.append(total_energy(N, masses, positions, velocities))
           momentum.append(total_momentum(masses, velocities))
           angular_momentum.append(total_angular_momentum(N, masses, positions, velocities))
           j.append(specific_angular_momentum(masses, positions, velocities))
           e.append(runge_lenz_vector(masses, positions, velocities))
           a.append(semimajor_axis(masses, positions, velocities))
           time.append(i * dt)
           (positions, velocities) = step_function(N, masses, positions, velocities, dt) 

        mom = np.linalg.norm(np.array(momentum), axis=1)
        angmom = np.linalg.norm(np.array(angular_momentum), axis=1)
        j = np.linalg.norm(np.array(j), axis=1)
        e = np.linalg.norm(np.array(e), axis=1)


        print(step_function.__name__)
        #plt.plot(time, energy, label='energy')
        #plt.plot(time, mom, label='momentum')
        #plt.plot(time, angmom, label='angular momentum')
        #plt.plot(time, a, label='semimajor axis')
        #plt.plot(time, j, label='specific angular momentum')
        #plt.plot(time, e, label='runge lenz vector')
        #plt.plot(time, (energy - energy[0])/energy[0], label='energy error')
        #plt.plot(time, (j - j[0])/j[0], label='specific ang mom error')
        plt.plot(time, (a - a[0])/a[0], label='semimajor axis error')

        plt.legend()
        plt.show()

two_body_test()
