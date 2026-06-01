import  numpy as np

class Planet:
    def __init__(self):
        pass

class Jupiter(Planet):
    def __init__(self):
        self.M = 1.898125e27
        self.R = 71488000

class Io(Planet):
    def __init__(self):
        self.M = 8.931938e22
        self.R = 71488000
        self.a = 420000000 
    
class Europa(Planet):
    def __init__(self):
        self.M = 4.79984e22
        self.R = 71488000
        self.a = 664862000

a = 420000000  # m #64862000
M_s = 8.931938e22 #kg
G = 6.67e-11
M_j = 1.898125e27 #lkg 
R_p = 71488000 #m
k2q = 1e-5 + 1e-6
k2q2 = 1e-5

def compute_TA_error(planet, moon, days = 365):
    R_over_a = planet.R / moon.a
    GM_overa3 = G * moon.M / moon.a ** 3
    drift = GM_overa3 * (R_over_a) ** 5 * 9/2
    print(drift * k2q2 * 60 * 60 * 24)
    days = days
    time_drift = -drift * (60 * 60 * 24 * days) ** 2
    time_drift = time_drift * 1e-6
    return time_drift

def get_required_accuracy(moon, TA_error):
    return np.abs(TA_error * moon.a)

def get_TA_erorrMS(moon, track_accuracy):
    return np.abs(track_accuracy / moon.a)


def get_required_days(moon, desired_accuracy):
    R_over_a = planet.R / moon.a
    GM_overa3 = G * moon.M / moon.a ** 3
    drift = GM_overa3 * (R_over_a) ** 5 * 9/2 * 1e-6
    time_drift = np.sqrt(desired_accuracy / (drift))
    return time_drift
planet = Jupiter()
moon = Europa()

result = compute_TA_error(planet, moon, 24.83)
print(result)
accuracy_along_track = 10 #m

error_true_anomaly = get_required_accuracy(moon, result)

print(error_true_anomaly)
time = get_required_days(moon, get_TA_erorrMS(moon, 0.1))
print(time /(60 * 60 * 24))
# n = (G * M_j / a ** 3) ** 0.5

# constant = 3 * k2q * M_s / M_j * (R_p / a) ** 5 * n * a * 60 * 60 * 24
# print(constant) # 3.24m => 100m / 30days

# constant2 = -9/2 * G * M_j * a ** -3 * k2q * M_s / M_j * (R_p / a) ** 5 
# constant3 = -9/2 * G * M_j * a ** -3 * k2q2 * M_s / M_j * (R_p / a) ** 5 
