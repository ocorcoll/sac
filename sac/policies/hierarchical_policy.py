import numpy as np
import tensorflow as tf

from contextlib import contextmanager
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from sac.misc.utils import concat_obs_z
from sac.policies import NNPolicy


class FixedOptionPolicy(object):
    def __init__(self, base_policy, num_skills, z):
        self._z = z
        self._base_policy = base_policy
        self._num_skills = num_skills

    def reset(self):
        pass

    def get_action(self, obs):
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)

    def get_distribution_for(self, obs_t, reuse=False):
        shape = [tf.shape(obs_t)[0]]
        z = tf.tile([self._z], shape)
        z_one_hot = tf.one_hot(z, self._num_skills, dtype=obs_t.dtype)
        aug_obs_t = tf.concat([obs_t, z_one_hot], axis=1)
        return self._base_policy.get_distribution_for(aug_obs_t, reuse=reuse)


class ScheduledOptionPolicy(object):
    def __init__(self, base_policy, num_skills, z_vec):
        self._z_vec = z_vec
        self._base_policy = base_policy
        self._num_skills = num_skills
        self._t = 0

    def reset(self):
        pass

    def get_action(self, obs):
        assert self._t < len(self._z_vec)
        z = self._z_vec[self._t]
        aug_obs = concat_obs_z(obs, z, self._num_skills)
        self._t += 1
        return self._base_policy.get_action(aug_obs)


class RandomOptionPolicy(object):

    def __init__(self, base_policy, num_skills, steps_per_option):
        self._num_skills = num_skills
        self._steps_per_option = steps_per_option
        self._base_policy = base_policy
        self.reset()

    def reset(self):
        self._z = np.random.choice(self._num_skills)

    def get_action(self, obs):
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)


class NNHierarchicalPolicy(NNPolicy, Serializable):

    def __init__(self, env_spec, policy):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Ds = env_spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Ds],
            name='observation',
        )

        self._output = policy.get_output_for(self._obs_pl, reuse=True)

        super(NNHierarchicalPolicy, self).__init__(
            env_spec,
            self._obs_pl,
            self._output,
            'policy'
        )


class HierarchicalPolicy(Serializable):
    def __init__(self, env_spec, base_policy, num_skills, meta_policy, steps_per_option):
        Serializable.quick_init(self, locals())
        self._steps_per_option = steps_per_option
        self._hierarchical_meta_policy = NNHierarchicalPolicy(env_spec, meta_policy)
        self._meta_policy = meta_policy
        self._base_policy = base_policy
        self._num_skills = num_skills
        self._is_deterministic = False
        self.reset()

    def reset(self):
        pass

    def get_action(self, obs):
        (z, _) = self._hierarchical_meta_policy.get_action(obs)
        z = np.argmax(z)
        aug_obs = concat_obs_z(obs, z, self._num_skills)
        return self._base_policy.get_action(aug_obs)

    def get_distribution_for(self, obs_t, reuse=False):
        z = self._meta_policy.get_output_for(obs_t, reuse=reuse)
        z = tf.nn.top_k(z, k=1).indices
        z_one_hot = tf.one_hot(z, self._num_skills, dtype=obs_t.dtype)
        z_one_hot = tf.squeeze(z_one_hot)

        aug_obs_t = tf.concat([obs_t, z_one_hot], axis=1)
        return self._base_policy.get_distribution_for(aug_obs_t, reuse=reuse)

    def get_params_internal(self, **tags):
        self._hierarchical_meta_policy.get_params_internal(**tags)

    @contextmanager
    def deterministic(self, set_deterministic=True):
        current = self._is_deterministic
        self._is_deterministic = set_deterministic
        yield
        self._is_deterministic = current

    @overrides
    def log_diagnostics(self, batch):
        z = self._hierarchical_meta_policy.get_actions(batch['observations'])
        top_z = np.argmax(z, axis=1)
        top_z = np.bincount(top_z)
        top_k = 5
        top_z = top_z.argsort()[-top_k:][::-1]
        logger.record_tabular('hierarchical-z', top_z)


class RandomHierarchicalPolicy(object):

    def __init__(self, base_policy, num_skills, steps_per_option):
        self._steps_per_option = steps_per_option
        self._base_policy = base_policy
        self._num_skills = num_skills
        self.reset()

    def reset(self):
        self._t = 0
        self._z = None

    def get_action(self, obs):
        # Choose a skill if necessary
        if self._t % self._steps_per_option == 0:
            self._z = np.random.choice(self._num_skills)
        self._t += 1
        aug_obs = concat_obs_z(obs, self._z, self._num_skills)
        return self._base_policy.get_action(aug_obs)
