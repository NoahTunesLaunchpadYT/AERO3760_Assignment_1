from Trajectory import Trajectory
from utils import calendar_to_jd

# ----------------- Simulator -----------------
class Simulator:
    def __init__(self, satellite, propagator):
        self.sat = satellite
        self.propagator = propagator

    def run(self, Y, M, D, h, m, s, tf_days, dt_seconds):
        jd0 = calendar_to_jd(Y, M, D, h, m, s)
        # propagate: returns arrays of elapsed seconds, positions, velocities
        t_s, R_km, V_km_s = self.propagator(self.sat.r0, self.sat.v0, 0, tf_days*86400, dt_seconds)
        jd_array = jd0 + t_s / 86400.0
        return Trajectory(jd_array, R_km, V_km_s, (Y, M, D, h, m, s))