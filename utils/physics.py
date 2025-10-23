import tensorflow as tf

# PRONOSTIA constants
p = 3.0
P = 4000.0
Ea = 0.1
kB = 8.617e-5
alpha = 1e-5
beta = 5e-6
gamma = 1.0
Dm = 0.025
K_archard = 1e-6
H_material = 1.5e9



# ------------------------------
# Physics-based degradation model
# ------------------------------


def K_arh(Load, RPM):
    return (RPM * (P ** p)) / (60e6 * (Load ** p))

def lmbda(T):
    return Ea / (kB * T)

def eta(Load, RPM):
    V_sliding = 3.14 * Dm * (RPM / 60)
    return (gamma * K_archard * Load * V_sliding) / H_material

def dDdt(t, Load, RPM, T):
    return K_arh(Load, RPM) * tf.exp((p * lmbda(T)) + (p * alpha * t)) * ((1 + eta(Load, RPM) * t) ** p)
