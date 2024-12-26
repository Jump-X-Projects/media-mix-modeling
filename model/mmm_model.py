# model/mmm_model.py
import pymc as pm
import pandas as pd
import numpy as np
import arviz as az
from scipy import stats

class MMMModel:
    def __init__(
        self,
        df,
        target_variable,
        date_column,
        media_channels,
        non_media_cols,
        n_draws,
        n_tune,
        n_chains,
        adstock_type,
        saturation_type,
        hyperparameters,
    ):
        self.df = df
        self.target_variable = target_variable
        self.date_column = date_column
        self.media_channels = media_channels
        self.non_media_cols = non_media_cols
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.adstock_type = adstock_type
        self.saturation_type = saturation_type
        self.hyperparameters = hyperparameters
        self.trace = None
        self.model = None

    def fit(self):
        """Fits the Bayesian MMM model."""
        with pm.Model() as self.model:
            mu_channel_contribs = []
            for channel in self.media_channels:
                if channel in self.df.columns:
                    # --- Adstock ---
                    if self.adstock_type == "geometric":
                        alpha = pm.Beta(
                            f"alpha_{channel}",
                            alpha=self.hyperparameters[channel]["Alpha (Lag)"] * 10,
                            beta=(1 - self.hyperparameters[channel]["Alpha (Lag)"])
                            * 10,
                        )
                        adstock = self._geometric_adstock(
                            self.df[channel].values, alpha=alpha, l_max=12, normalize=True
                        )

                    elif self.adstock_type == "delayed":
                        alpha = pm.Beta(
                            f"alpha_{channel}",
                            alpha=self.hyperparameters[channel]["Alpha (Lag)"] * 10,
                            beta=(1 - self.hyperparameters[channel]["Alpha (Lag)"])
                            * 10,
                        )
                        gamma = pm.Beta(
                            f"gamma_{channel}",
                            alpha=self.hyperparameters[channel]["Gamma (Decay)"] * 10,
                            beta=(1 - self.hyperparameters[channel]["Gamma (Decay)"])
                            * 10,
                        )
                        adstock = self._delayed_adstock(
                            self.df[channel].values,
                            alpha=alpha,
                            gamma=gamma,
                            l_max=12,
                            normalize=True,
                        )
                    else:
                        raise ValueError(
                            f"Invalid adstock_type: {self.adstock_type}. Must be 'geometric' or 'delayed'"
                        )

                    # --- Saturation ---
                    if self.saturation_type == "hill":
                        beta = pm.HalfNormal(
                            f"beta_{channel}",
                            sigma=self.hyperparameters[channel]["Beta (Saturation)"],
                        )
                        lam = pm.HalfNormal(
                            f"lam_{channel}",
                            sigma=self.hyperparameters[channel]["Beta (Saturation)"],
                        )
                        sat_adstock = self._hill_saturation(adstock, lam, beta)
                        self.df[f"{channel}_adstocked_saturated"] = sat_adstock.eval()

                    elif self.saturation_type == "reach":
                        beta = pm.Beta(
                            f"beta_{channel}",
                            alpha=self.hyperparameters[channel]["Beta (Saturation)"] * 10,
                            beta=(1 - self.hyperparameters[channel]["Beta (Saturation)"])
                            * 10,
                        )
                        sat_adstock = self._reach_saturation(adstock, beta)
                        self.df[f"{channel}_adstocked_saturated"] = sat_adstock.eval()

                    else:
                        raise ValueError(
                            f"Invalid saturation_type: {self.saturation_type}. Must be 'hill' or 'reach'"
                        )

                    coef = pm.HalfNormal(f"coef_{channel}", sigma=2)

                    mu_channel_contrib = pm.Deterministic(
                        f"{channel}_contribution", coef * sat_adstock
                    )
                    mu_channel_contribs.append(mu_channel_contrib)

            # --- Control Variables ---
            control_coefs = []
            for i, var in enumerate(self.non_media_cols):
                if var in self.df.columns:
                  coef = pm.Normal(f"coef_control_{i}", mu=0, sigma=2)
                  control_coefs.append(coef)
                  mu_channel_contribs.append(coef * self.df[var].values)

            # --- Intercept ---
            intercept = pm.Normal("intercept", mu=0, sigma=2)

            # --- Likelihood ---
            sigma = pm.HalfNormal("sigma", sigma=2)
            mu = intercept + sum(mu_channel_contribs)
            pm.Normal(
                "target",
                mu=mu,
                sigma=sigma,
                observed=self.df[self.target_variable].values,
            )

            # --- Sampling ---
            self.trace = pm.sample(
                self.n_draws,
                tune=self.n_tune,
                chains=self.n_chains,
                target_accept=0.95,
                return_inferencedata=True,
                cores=4
            )

            # --- Posterior Predictive Checks ---
            pm.sample_posterior_predictive(
                self.trace,
                var_names=[
                    "target",
                    "intercept",
                ]
                + [f"{channel}_contribution" for channel in self.media_channels]
                + [f"coef_{channel}" for channel in self.media_channels],
                extend_inferencedata=True,
            )
            self.df["target_prediction"] = self.trace.posterior_predictive[
                "target"
            ].mean(dim=["chain", "draw"])

            # --- Calculate Channel Contributions ---
            for channel in self.media_channels:
                self.df[f"{channel}_{self.target_variable}_contribution"] = (
                    self.trace.posterior_predictive[f"{channel}_contribution"]
                    .mean(dim=["chain", "draw"])
                    .to_numpy()
                )

            # --- Calculate Response Curves ---
            max_spend = self.df[self.media_channels].max().to_dict()
            mean_spend = self.df[self.media_channels].mean().to_dict()
            std_spend = self.df[self.media_channels].std().to_dict()

            for channel in self.media_channels:
                if channel in max_spend:
                    max_val = max_spend[channel]
                    min_val = max_spend[channel] * 0.05
                    if max_val == 0:
                        max_val = mean_spend[channel] + 2 * std_spend[channel]

                    # Create a range of spend values for the channel
                    spend_range = np.linspace(
                        min_val, max_val, 100
                    )

                    # Calculate the response curve for the channel using the posterior samples
                    response_curves = []
                    for i in range(self.trace.posterior.dims["draw"]):
                      if self.adstock_type == "geometric":
                        adstock_effect = self._geometric_adstock(
                            spend_range,
                            alpha=self.trace.posterior[f"alpha_{channel}"].values[0, i],
                            l_max=12,
                            normalize=True,
                        )
                      elif self.adstock_type == "delayed":
                        adstock_effect = self._delayed_adstock(
                          spend_range,
                          alpha=self.trace.posterior[f"alpha_{channel}"].values[0, i],
                          gamma=self.trace.posterior[f"gamma_{channel}"].values[0, i],
                          l_max=12,
                          normalize=True,
                        )

                      if self.saturation_type == "hill":
                        saturation_effect = self._hill_saturation(
                          adstock_effect,
                          lam=self.trace.posterior[f"lam_{channel}"].values[0, i],
                          beta=self.trace.posterior[f"beta_{channel}"].values[0, i],
                        )

                      elif self.saturation_type == "reach":
                        saturation_effect = self._reach_saturation(
                          adstock_effect,
                          beta=self.trace.posterior[f"beta_{channel}"].values[0, i],
                        )
                      response_curve = (
                          self.trace.posterior[f"coef_{channel}"].values[0, i]
                          * saturation_effect
                      )
                      response_curves.append(response_curve)

                    # Calculate the mean response curve for the channel
                    mean_response_curve = np.mean(response_curves, axis=0)

                    # Store the response curve in the dataframe
                    self.df[f"{channel}_{self.target_variable}_response_curve"] = np.interp(
                      self.df[channel], spend_range, mean_response_curve
                    )

    def _geometric_adstock(self, x, alpha, l_max, normalize=False):
        w = np.array([alpha**i for i in range(l_max)])
        if normalize:
            w = w / np.sum(w)
        return np.convolve(x, w, mode="full")[: len(x)]

    def _delayed_adstock(self, x, alpha, gamma, l_max, normalize=False):
        w = np.array(
            [
                alpha * stats.beta.cdf(i + 1, alpha, gamma)
                - alpha * stats.beta.cdf(i, alpha, gamma)
                for i in range(l_max)
            ]
        )
        if normalize:
            w = w / np.sum(w)
        return np.convolve(x, w, mode="full")[: len(x)]

    def _hill_saturation(self, x, lam, beta):
        return x**beta / (lam**beta + x**beta)

    def _reach_saturation(self, x, beta):
        return 1 - np.exp(-beta * x)

    def get_coefficients(self):
        """Returns the model coefficients (betas)."""
        coef_df = pd.DataFrame()
        for channel in self.media_channels:
            if channel in self.df.columns:
                coef_df[channel] = self.trace.posterior[f"coef_{channel}"].mean(
                    dim=["chain", "draw"]
                )
        coef_df = coef_df.rename(
            columns={
                channel: f"{channel}_coef"
                for channel in self.media_channels
                if channel in self.df.columns
            }
        )
        return coef_df

    def get_media_contribution(self):
        """Returns the media channel contributions."""
        media_contributions = pd.DataFrame()
        for channel in self.media_channels:
            if channel in self.df.columns:
                media_contributions[
                    f"{channel}_{self.target_variable}_contribution"
                ] = self.df[f"{channel}_{self.target_variable}_contribution"]
        media_contributions.index = self.df[self.date_column]
        return media_contributions

    def get_response_curves(self):
        """Returns the response curves."""
        response_curves_df = pd.DataFrame()
        for channel in self.media_channels:
            if channel in self.df.columns:
                response_curves_df[
                    f"{channel}_{self.target_variable}_response_curve"
                ] = self.df[f"{channel}_{self.target_variable}_response_curve"]
        response_curves_df.index = self.df[self.date_column]
        return response_curves_df