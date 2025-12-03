import unittest
import numpy as np
import jax
import jax.numpy as jnp

from slimsac.algorithms.sac import SAC
from tests.utils import Generator


class TestSAC(unittest.TestCase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_seed = np.random.randint(1000)
        self.key = jax.random.PRNGKey(self.random_seed)

        key_obs_dim, key_actions, key_feature_pi_1, key_feature_pi_2, key_feature_q_1, key_feature_q_2 = (
            jax.random.split(self.key, 6)
        )
        self.observation_dim = (int(jax.random.randint(key_obs_dim, (), minval=5, maxval=10)),)
        self.n_actions = int(jax.random.randint(key_actions, (), minval=2, maxval=10))
        self.q = SAC(
            self.key,
            self.observation_dim,
            self.n_actions,
            0.001,
            0.94,
            1,
            0.5,
            [
                jax.random.randint(key_feature_pi_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_pi_2, (), minval=1, maxval=10),
            ],
            [
                jax.random.randint(key_feature_q_1, (), minval=1, maxval=10),
                jax.random.randint(key_feature_q_2, (), minval=1, maxval=10),
            ],
        )

        self.generator = Generator(10, self.observation_dim, self.n_actions)

    def test_compute_target(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        next_actions, next_log_probs = self.q.actor.apply(self.q.actor_params, samples.next_state, self.key)
        next_q_values_double = jax.vmap(self.q.critic.apply, in_axes=(0, None, None))(
            self.q.critic_target_params, samples.next_state, next_actions
        )
        next_q_values = jnp.min(next_q_values_double, axis=0)
        computed_target = self.q.compute_target(samples, next_q_values, jnp.exp(self.q.log_ent_coef), next_log_probs)

        target = samples.reward + (1 - samples.is_terminal) * (self.q.gamma**self.q.update_horizon) * (
            next_q_values - jnp.exp(self.q.log_ent_coef) * next_log_probs
        )

        np.testing.assert_array_almost_equal(target, computed_target)

    def test_critic_loss_on_batch(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        next_actions, next_log_probs = self.q.actor.apply(self.q.actor_params, samples.next_state, self.key)
        q_values = jax.vmap(self.q.critic.apply, in_axes=(0, None, None))(
            self.q.critic_params, samples.state, samples.action
        )
        next_q_values_double = jax.vmap(self.q.critic.apply, in_axes=(0, None, None))(
            self.q.critic_target_params, samples.next_state, next_actions
        )
        next_q_values = jnp.min(next_q_values_double, axis=0)
        targets_ = self.q.compute_target(samples, next_q_values, jnp.exp(self.q.log_ent_coef), next_log_probs)
        targets = jnp.repeat(targets_[jnp.newaxis], 2, axis=0)
        critic_loss = jnp.square(q_values - targets).mean()

        computed_critic_loss = self.q.critic_loss_on_batch(
            self.q.critic_params,
            self.q.critic_target_params,
            self.q.actor_params,
            self.q.log_ent_coef,
            samples,
            self.key,
        )
        self.assertEqual(critic_loss, computed_critic_loss)

    def test_actor_loss_on_batch(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")
        samples = self.generator.samples(self.key)

        actions, log_probs = self.q.actor.apply(self.q.actor_params, samples.state, self.key)
        q_values_double = jax.vmap(self.q.critic.apply, in_axes=(0, None, None))(
            self.q.critic_params, samples.state, actions
        )
        q_values = jnp.min(q_values_double, axis=0)
        losses = jnp.exp(self.q.log_ent_coef) * log_probs - q_values

        computed_actor_loss_terms = self.q.actor_loss_on_batch(
            self.q.actor_params, self.q.critic_params, self.q.log_ent_coef, samples, self.key
        )
        self.assertEqual(losses.mean(), computed_actor_loss_terms[0])
        self.assertEqual(-log_probs.mean(), computed_actor_loss_terms[1])

    def test_entropy_loss(self) -> None:
        print(f"-------------- Random key {self.random_seed} --------------")

        entropy = -jnp.log(jax.random.uniform(self.key))
        entropy_loss = jnp.exp(self.q.log_ent_coef) * (entropy - self.q.target_entropy)
        computed_entropy_loss = self.q.entropy_loss(self.q.log_ent_coef, entropy)

        self.assertEqual(entropy_loss, computed_entropy_loss)

    def test_sample_action(self):
        print(f"-------------- Random key {self.random_seed} --------------")
        state = self.generator.state(self.key)

        computed_action = self.q.sample_action(state, self.q.actor_params, self.key)

        action, _ = self.q.actor.apply(self.q.actor_params, state, self.key)

        self.assertEqual(computed_action.shape, (self.n_actions,))
        np.testing.assert_array_almost_equal(computed_action, action)
