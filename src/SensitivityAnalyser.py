import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from OrbitDetermination import SatelliteLaserRanging, RadiometricTracking, OpticalTracking, OrbitDeterminationSolver

# Treat warnings as errors
import warnings
warnings.simplefilter("error", RuntimeWarning)

# Types
ErrorType = Literal["angular", "temporal", "range"]
MethodType = Literal["Gibbs", "Lambert", "Gauss"]


@dataclass
class SensitivityResult:
    errors: List[float]
    percent_errors: Dict[MethodType, np.ndarray]  # shape: (len(errors), 6)
    labels: Tuple[str, str, str, str, str, str] = (
        "a (km)", "e", "i (deg)", "RAAN (deg)", "AOP (deg)", "TA (deg)"
    )


class SensitivityAnalyser:
    """
    Runs sensor-error sensitivity sweeps and compares OD methods.

      • 3 observation times provided as Julian dates; Lambert uses the first two only.
      • Three sensor types are available: SatelliteLaserRanging, RadiometricTracking, OpticalTracking.
      • OrbitDeterminationSolver.determine_orbit(gs, observables=..., method=..., trajectory=optional)
      • Satellite-like objects implement .keplerian_elements() -> (a_km,e,i_deg,raan_deg,aop_deg,ta_deg)
    """

    KEPLER_LABELS = ("a (km)", "e", "i (deg)", "RAAN (deg)", "AOP (deg)", "TA (deg)")

    def __init__(
        self,
        sim,
        sat_key: str,
        gs_key: str,
        od_solver,
        true_elements: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> None:
        self.sim = sim
        self.sat_key = sat_key
        self.gs_key = gs_key
        self.od_solver = od_solver

        # Fetch true elements from the reference satellite if not provided
        if true_elements is None:
            sat = self.sim.satellites[self.sat_key]
            true_elements = sat.keplerian_elements()
        self.true_elements = true_elements  # (a,e,i,RAAN,AOP,TA)
        
        # Convenience handles
        self.gs = self.sim.ground_stations[self.gs_key]
        self.sat_traj = self.sim.satellite_trajectories[self.sat_key]
        self.gs_traj = self.sim.ground_station_trajectories[self.gs_key]

    # ------------------------------ Public API ------------------------------ #
    def sensor_error_vs_predict_error(
        self,
        observation_times_jd: Tuple[float, float, float],
        error_type: ErrorType,
        error_values: Sequence[float],
        methods: Sequence[MethodType] = ("Gibbs", "Lambert", "Gauss"),
        repeats: float = 20,
        param_mask: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        ylog: bool = False,
        ab_error_instead: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]] = None,
        block = True,
        plot_obs = False
    ) -> SensitivityResult:
        """
        Sweep a single sensor error dimension and plot percent element errors.

        Parameters
        ----------
        observation_times_jd : (t1, t2, t3)
            Three observation times (JD). Lambert uses (t1, t2).
        error_type : {'angular','temporal','range'}
            Which sensor error to sweep.
        error_values : sequence of floats
            Magnitudes to try (units match the chosen error_type in sensor classes).
        methods : subset of {'Gibbs','Lambert','Gauss'}
            OD methods to evaluate and plot.
        ylog : bool
            Plot y-axis on log scale (useful if errors span many orders of magnitude).
        safe_denoms : tuple of 6 optional scales
            If a true value is 0 (e.g. e = 0, RAAN = 0), the percent error is undefined.
            Provide a replacement denominator (e.g. 1 deg) per element: (a,e,i,RAAN,AOP,TA).
            If None, NaNs will appear where division is undefined.
        """
        t1, t2, t3 = observation_times_jd
        errors = list(error_values)

        # Results per method: array shape (len(errors), 6)
        per_method = {m: np.zeros((len(errors), 6), dtype=float) for m in methods}

        from tqdm import tqdm

        for j, err in enumerate(tqdm(errors, desc=f"Sweeping {error_type} error", unit="val")):
            # Accumulators for averaging across repeats
            accum = {m: np.zeros(6, dtype=float) for m in methods}

            for i in range(repeats):
                # Build sensors for the current error magnitude in the swept dimension
                range_sensor, radio_sensor, optical_sensor = self._build_sensors(err, error_type)

                # Collect observations
                range_obs = range_sensor.observe(self.sat_traj, self.gs_traj, jd_times=(t1, t2, t3))

                # Solve with requested methods and accumulate percent errors
                if "Gibbs" in methods:
                    range_obs = range_sensor.observe(self.sat_traj, self.gs_traj, jd_times=(t1, t2, t3))
                    sat_gibbs = self.od_solver.determine_orbit(self.gs, observables=range_obs, method="Gibbs", plot=plot_obs)
                    accum["Gibbs"] += self._percent_element_errors(sat_gibbs, ab_error_instead=ab_error_instead)
                    if plot_obs and "Gauss" not in methods:
                        plot_obs = False

                if "Lambert" in methods:
                    sat_lambert = None
                    while sat_lambert is None:
                        radio_obs = radio_sensor.observe(self.sat_traj, self.gs_traj, jd_times=(t1, t2))
                        sat_lambert = self.od_solver.determine_orbit(self.gs, observables=radio_obs, method="Lambert")
                    
                    accum["Lambert"] += self._percent_element_errors(sat_lambert, ab_error_instead=ab_error_instead)

                if "Gauss" in methods:
                    sat_gauss = None
                    while sat_gauss is None:
                        optical_obs = optical_sensor.observe(self.sat_traj, self.gs_traj, jd_times=(t1, t2, t3))
                        sat_gauss = self.od_solver.determine_orbit(
                            self.gs, observables=optical_obs, method="Gauss", trajectory=self.sat_traj, plot=plot_obs
                        )
                    if plot_obs:
                        plot_obs = False

                    accum["Gauss"] += self._percent_element_errors(sat_gauss, ab_error_instead=ab_error_instead)

            # Average over repeats
            for m in methods:
                per_method[m][j, :] = accum[m] / float(repeats)

        # Optional plotting
        self._plot_sweep(
            errors,
            per_method,
            error_type,
            param_mask=param_mask,
            figsize=figsize,
            ylog=ylog,
            ab_error_instead=ab_error_instead,
            block=block,
        ) if show else None

        return SensitivityResult(errors=errors, percent_errors=per_method, labels=self.KEPLER_LABELS)

    # ------------------------------ Internals ------------------------------ #
    def _build_sensors(self, err_mag: float, error_type: ErrorType):
        # Default all errors to zero then set one dimension to err_mag
        def errs():
            if error_type == "range":
                return dict(range_error=err_mag, angular_error=0.0, time_error=0.0)
            if error_type == "angular":
                return dict(range_error=0.0, angular_error=err_mag, time_error=0.0)
            if error_type == "temporal":
                return dict(range_error=0.0, angular_error=0.0, time_error=err_mag)
            if error_type == "all":
                return dict(range_error=err_mag, angular_error=err_mag, time_error=err_mag)
            raise ValueError("Unknown error_type")

        e = errs()
        range_sensor = SatelliteLaserRanging(id="range_sensor", **e)
        radio_sensor = RadiometricTracking(id="radio_sensor", **e)
        optical_sensor = OpticalTracking(id="optical_sensor", **e)
        return range_sensor, radio_sensor, optical_sensor

    def _percent_element_errors(self, predicted_sat, ab_error_instead: Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]] | None = None) -> np.ndarray:
        pred = predicted_sat.keplerian_elements()  # (a,e,i,RAAN,AOP,TA)
        true = self.true_elements
        pred = np.asarray(pred, dtype=float)
        true = np.asarray(true, dtype=float)

        perc = []
        
        if ab_error_instead is None:
            for i in range(len(true)):
                perc.append(np.abs(pred[i] - true[i]) / np.abs(true[i]) * 100.0)
        else:
            for i in range(len(ab_error_instead)):
                if ab_error_instead[i]:
                    perc.append(abs(pred[i] - true[i]))
                else:
                    perc.append(np.abs(pred[i] - true[i]) / np.abs(true[i]) * 100.0)

        return perc

    def _plot_sweep(
        self,
        x_errors: Sequence[float],
        per_method: Dict[MethodType, np.ndarray],
        error_type: ErrorType,
        param_mask: Optional[Sequence[bool]] = None,
        figsize: Tuple[int, int] = (8, 12),
        ylog: bool = False,
        ab_error_instead: Optional[Sequence[bool]] = None,
        block: bool = True
    ) -> None:
        """Create a vertically stacked set of subplots (aligned x-axis), lines per method.

        param_mask selects which orbital elements to plot in order (a, e, i, RAAN, AOP, TA).
        Example: [1, 0, 0, 0, 0, 0] plots only semi-major axis.
        """
        x = np.asarray(x_errors, dtype=float)
        x = x*100  # Convert to percentage

        # Build list of indices to plot
        if param_mask is None:
            indices = list(range(6))
        else:
            if len(param_mask) != 6:
                raise ValueError("param_mask must have length 6 (a, e, i, RAAN, AOP, TA)")
            indices = [i for i, keep in enumerate(param_mask) if bool(keep)]
            if not indices:
                # Nothing to plot; just return
                return

        nrows = len(indices)
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True, constrained_layout=True)
        if nrows == 1:
            axes = [axes]

        axes[0].set_title(f"{error_type.capitalize()} Error vs Orbital Parameter Error")

        for ax, idx in zip(axes, indices):
            label = self.KEPLER_LABELS[idx]
            for method, mat in per_method.items():
                y = mat[:, idx]
                ax.plot(x, y, marker="", label=method)

            if ab_error_instead and ab_error_instead[idx]:
                ax.set_ylabel(f"Abs err in: {label}")
            else:
                ax.set_ylabel(f"% err in: {label}")
                ax.axhline(100, color="red", ls="--", label="100% error")
            if ylog:
                ax.set_yscale("log")
            ax.grid(True, which="both", ls=":", alpha=0.6)
            # ax.set_title(label)

        # Only bottom plot gets the x label for cleaner stacking
        axes[-1].set_xlabel(f"{error_type} error %")

        # Put a single legend on the top axis
        axes[0].legend(loc="best")
        plt.show(block=block)
