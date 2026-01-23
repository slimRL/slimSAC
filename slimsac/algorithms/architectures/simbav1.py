import flax.linen as nn
import jax.numpy as jnp
import jax


class SimbaV1CriticNet(nn.Module):
    feature: int

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = RSObservationNorm()(jnp.squeeze(state))
        x = jnp.concatenate([x, action.astype(jnp.float32)], -1)
        x = nn.Dense(self.feature, kernel_init=nn.initializers.orthogonal(1.0))(x)

        for _ in range(2):
            res = x
            x = nn.LayerNorm()(x)
            x = nn.Dense(self.feature * 4, kernel_init=nn.initializers.he_normal())(x)
            x = nn.relu(x)
            x = nn.Dense(self.feature, kernel_init=nn.initializers.he_normal())(x)
            x = res + x
        x = nn.LayerNorm()(x)

        return nn.Dense(features=1, kernel_init=nn.initializers.orthogonal(1.0))(x)


class SimbaV1ActorNet(nn.Module):
    feature: int
    action_dim: int
    min_log_stds = -10
    max_log_stds = 2

    @nn.compact
    def __call__(self, state: jnp.ndarray, noise_key) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = RSObservationNorm()(jnp.squeeze(state))
        x = nn.Dense(self.feature, kernel_init=nn.initializers.orthogonal(1.0))(x)
        res = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.feature * 4, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(self.feature, kernel_init=nn.initializers.he_normal())(x)
        x = res + x
        x = nn.LayerNorm()(x)

        means = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(1.0))(x)

        if noise_key is None:  # deterministic
            return jnp.tanh(means), None
        else:
            log_stds_unclipped = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(1.0))(x)
            # Apply tanh to prediction (output in [-1, 1]) and then scale (output in [min_log_stds, max_log_stds])
            log_stds = self.min_log_stds + (self.max_log_stds - self.min_log_stds) / 2 * (
                1 + nn.tanh(log_stds_unclipped)
            )
            stds = jnp.exp(log_stds)

            action_pre_tanh = means + stds * jax.random.normal(noise_key, shape=stds.shape, dtype=jnp.float32)
            action = jnp.tanh(action_pre_tanh)

            # Gaussian log-prob: -1/2 ((x - mean) / std)^2 -1/2 log(2 pi) -log(sigma)
            log_prob_uncorrected = (
                -0.5 * jnp.square(action_pre_tanh / stds - means / stds) - 0.5 * jnp.log(2 * jnp.pi) - log_stds
            )
            # d tanh^{-1}(y) / dy = 1 / (1 - y^2)
            log_prob = log_prob_uncorrected - jnp.log(1 - action**2 + 1e-6)

            return action, jnp.sum(log_prob, axis=-1)


class RSObservationNorm(nn.Module):
    @nn.compact
    def __call__(self, x) -> jax.Array:
        mean = self.variable("running_obs_stats", "mean", lambda: jnp.zeros((x.shape[-1],)))
        var = self.variable("running_obs_stats", "var", lambda: jnp.ones((x.shape[-1],)))
        count = self.variable("running_obs_stats", "count", lambda: jnp.array(1.0))

        mean = jax.lax.stop_gradient(mean.value)
        var = jax.lax.stop_gradient(var.value)
        norm64 = (x - mean) / jnp.sqrt(var + 1e-8)
        return norm64.astype(jnp.float32)


@jax.jit
def update_mean_var_stats(x, stats):
    return {
        # \mu_t = \mu_{t-1} + 1 / t * (o_t - \mu_{t-1})
        "mean": stats["mean"] + 1 / stats["count"] * (x - stats["mean"]),
        # \sigma_t = (t - 1) / t * [\sigma_{t-1} + 1 / t * (o_t - \mu_{t-1})^2]
        "var": (stats["count"] - 1) / stats["count"] * (stats["var"] + jnp.square(x - stats["mean"]) / stats["count"]),
        "count": stats["count"] + 1,
    }
