from scipy.integrate import solve_ivp
import numpy as np
from j2 import *
from utils import *

# ----------------- Propagator (SciPy) -----------------
class PropagatorSciPy:
    def __init__(self, method="DOP853", rtol=1e-8, atol=1e-10, max_step=np.inf):
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

    def propagate(self, r0, v0, t0, tf, sample_dt=None):
        y0 = np.hstack([np.array(r0, float), np.array(v0, float)])
        sol = solve_ivp(rhs_j2, (t0, tf), y0,
                        method=self.method,
                        rtol=self.rtol, atol=self.atol,
                        max_step=self.max_step,
                        dense_output=(sample_dt is not None))
        if not sol.success:
            raise RuntimeError("Propagation failed: " + sol.message)

        if sample_dt is None:
            T = sol.t
            Y = sol.y.T
        else:
            T = np.arange(t0, tf + 1e-9, sample_dt)
            Y = sol.sol(T).T

        R = Y[:, 0:3]
        V = Y[:, 3:6]
        return T, R, V