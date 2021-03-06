import os
import tempfile
import os.path as osp
import pickle

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from tqdm import tqdm


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so we ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)

    def save(self, path):
        save_variables(path)


def load_act(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load_act(path)


def learn(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=1000000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=10,#100
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=50000,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          ddqn=False,
          prior=False,
          save_freq=True,
          save_freq_rate=1000000,
          **network_kwargs
          ):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    param_noise: bool
        whether or not to use parameter space noise (https://arxiv.org/abs/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    checkpoint_path = logger.get_dir()
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    blr_params = BLRParams()
    # q_func = build_q_func(network, **network_kwargs)
    q_func = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[blr_params.feat_dim],
        dueling=bool(0),
        neural_linear=True
    )
    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug, feat, blr_ops = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        double_q=ddqn
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "best_model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))


        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # Take action and update exploration to the newest value
            # if t % 10000 == 0:
            #     print("{}/{}".format(t,total_timesteps))
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                print("updateing target, steps {}".format(t))
                update_target()

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)
            num_episodes = len(episode_rewards)
            # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            if t % 1000 == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 10 episode reward", mean_100ep_reward)
                logger.dump_tabular()
            if save_freq:
                if t > 0 and t % save_freq_rate == 0:
                    print("saving model periodically")
                    temp_model_file = os.path.join(checkpoint_path, "model_{}".format(t // checkpoint_freq))
                    save_variables(temp_model_file)
                    phiphiT, phiY = calculate_precision(replay_buffer, env.action_space.n, blr_ops, blr_params, n_samples=100000)
                    print("saving data to:")
                    print(osp.join(checkpoint_path,"phiphiT_{}.pickle".format(t // checkpoint_freq)))
                    with open(osp.join(checkpoint_path,"phiphiT_{}.pickle".format(t // checkpoint_freq)),'wb') as f:
                        pickle.dump(phiphiT,f)
                    with open(osp.join(checkpoint_path,"phiY_{}.pickle".format(t // checkpoint_freq)),'wb') as f:
                        pickle.dump(phiphiT,f)

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act

from tqdm import tqdm
import cvxpy as cvx
from scipy.stats import invgamma

class BLRParams(object):
    def __init__(self):
        self.sigma = 0.001 #W prior variance
        self.sigma_n = 1 # noise variance
        self.alpha = .01 # forgetting factor
        self.sample_w = 1000
        self.batch_size = 100000# batch size to do blr from
        self.gamma = 0.99 #dqn gamma
        self.feat_dim = 512 #64
        self.first_time = True
        self.no_prior = True
        self.a0 = 6
        self.b0 = 6

def information_transfer(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, batch_size, num_actions, feat_dim):
    d = [[] for i in range(num_actions)]
    phi = [[] for i in range(num_actions)]
    n = [0 for i in range(num_actions)]
    print("transforming information")
    from datetime import datetime
    fmt = '%Y-%m-%d %H:%M:%S'
    d1 = datetime.strptime('2010-01-01 17:31:22', fmt)
    for j in range(batch_size):
        obs_t, action, reward, obs_tp1, done = replay_buffer.sample(1)

        # confidence scores for old data
        c = target_dqn_feat(obs_tp1).reshape((feat_dim, 1))

        d[int(action)].append(np.dot(np.dot(c.T, phiphiT[int(action)]), c))


        # new data correlations
        c = dqn_feat(obs_tp1).reshape((feat_dim, 1))
        phi[int(action)].append(np.outer(c, c))
        n[int(action)] += 1
    print(n,sum(n))
    precisions_return = []
    cov = []
    prior = (0.001) * np.eye(feat_dim)
    print("solving optimization")
    for a in range(num_actions):
        print("for action {}".format(a))
        if d[a] != []:
            X = cvx.Variable((feat_dim, feat_dim), PSD=True)
            # Form objective.
            obj = cvx.Minimize(sum([(cvx.trace(X * phi[a][i]) - np.squeeze(d[a][i])) ** 2 for i in range(len(d[a]))]))
            prob = cvx.Problem(obj)
            prob.solve()
            if X.value is None:
                print("failed - cvxpy couldn't solve for action {}".format(a))
                precisions_return.append(np.linalg.inv(prior))
                cov.append(prior)
            else:
                precisions_return.append(np.linalg.inv(X.value + prior))
                cov.append(X.value + prior)
        else:
            print("failed - no samples for this action - {}".format(a))
            precisions_return.append(np.linalg.inv(prior))
            cov.append(prior)
    d2 = datetime.strptime('2010-01-03 17:31:22', fmt)
    print("total time for information transfer:")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    return precisions_return, cov

def calculate_precision(replay_buffer, num_actions, blr_ops, blr_params, n_samples=100000):
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(n_samples)
                        mini_batch_size = 32*num_actions
                        phiphiT = np.zeros((num_actions, blr_params.feat_dim, blr_params.feat_dim))
                        phiY = np.zeros((num_actions, blr_params.feat_dim))
                        n = [0 for _ in range(num_actions)]
                        for j in tqdm(range((n_samples // mini_batch_size)+1)):
                            start_idx = j*mini_batch_size
                            end_idx = (j+1)*mini_batch_size if (j+1)*mini_batch_size < len(replay_buffer) else -1
                            obs_t = obses_t[start_idx:end_idx] / 255.
                            action = actions[start_idx:end_idx]
                            reward = rewards[start_idx:end_idx]
                            obs_tp1 = obses_tp1[start_idx:end_idx] / 255.
                            done = dones[start_idx:end_idx]

                            # calculating target
                            for k in range(num_actions):
                                if obs_t[action == k].shape[0] < 1:
                                    continue
                                phiphiTk, phiYk = blr_ops(obs_t[action == k], action[action == k], reward[action == k], obs_tp1[action == k], done[action == k])
                                phiphiT[k] += phiphiTk
                                phiY[k] += phiYk
                                n[k] += obs_t[action == k].shape[0]
                        return phiphiT, phiY

def calc_precision_prior(self,contexts, blr_params, phiphiT):
    precisions_return = []
    n,m = contexts.shape
    prior = (blr_params.sigma) * np.eye(blr_params.feat_dim)

    if self.cov is not None:
      for action,precision in enumerate(self.cov):
        ind = np.array([i for i in range(n) if self.data_h.actions[i] == action])
        if len(ind)>0:

          """compute confidence scores for old data"""
          d = []
          for c in self.latent_h.contexts[ind, :]:
            d.append(np.dot(np.dot(c,precision),c.T))

          """compute new data correlations"""
          phi = []
          for c in contexts[ind, :]:
            phi.append(np.outer(c,c))

          X = cvx.Variable((m, m), PSD=True)
          # Form objective.
          obj = cvx.Minimize(sum([(cvx.trace(X*phi[i]) - d[i])**2 for i in xrange(len(d))]))
          prob = cvx.Problem(obj)
          prob.solve()
          if X.value is None:
            precisions_return.append(np.linalg.inv(prior))
            self.cov[action] = prior

          else:
            precisions_return.append(np.linalg.inv(X.value+prior))
            self.cov[action] = X.value+prior
        else:
          precisions_return.append(np.linalg.inv(prior))
          self.cov[action] = prior

    return precisions_return

def information_transfer_old(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, batch_size, num_actions, feat_dim):
    d = [[] for i in range(num_actions)]
    phi = [[] for i in range(num_actions)]
    for j in range(batch_size):
        obs_t, action, reward, obs_tp1, done = replay_buffer.sample(1)

        # confidence scores for old data
        c = target_dqn_feat(obs_tp1).reshape((feat_dim, 1))

        d[int(action)].append(np.dot(np.dot(c.T, phiphiT[int(action)]), c))


        # new data correlations
        c = dqn_feat(obs_tp1).reshape((feat_dim, 1))
        phi[int(action)].append(np.outer(c, c))

    precisions_return = []
    cov = []
    print("solving optimization")
    prior = (0.001) * np.eye(feat_dim)
    for a in range(num_actions):
        print("for action {}".format(a))
        if d[a] != []:
            X = cvx.Variable((feat_dim, feat_dim), PSD=True)
            # Form objective.
            obj = cvx.Minimize(sum([(cvx.trace(X * phi[a][i]) - np.squeeze(d[a][i])) ** 2 for i in range(len(d[a]))]))
            prob = cvx.Problem(obj)
            prob.solve()
            if X.value is None:
                precisions_return.append(np.linalg.inv(prior))
                cov.append(prior)
            else:
                precisions_return.append(np.linalg.inv(X.value + prior))
                cov.append(X.value + prior)
        else:
            precisions_return.append(np.linalg.inv(prior))
            cov.append(prior)
    d2 = datetime.strptime('2010-01-03 17:31:22', fmt)
    print("total time for information transfer:")
    print(d2.minute - d1.minute + (d2.hour-d1.hour) * 60)
    return precisions_return, cov

def BayesRegNoPrior(phiphiT, phiY, w_target, replay_buffer,dqn_feat, target_dqn_feat, target, num_actions, blr_param,
                    w_mu, w_cov, last_layer_weights, prior="no prior", blr_ops=None, blr_helpers=None):
    # dqn_ and target_dqn_ are feature extractors for the dqn and target dqn respectivley
    # in the neural linear settings target are the old features and dqn are the new features
    # last_layer_weights are the target last layer weights
    gamma = blr_param.gamma
    feat_dim = blr_param.feat_dim

    if prior == "no prior":
        print("no prior")
        phiY *= (1-blr_param.alpha)*0
        phiphiT *= (1-blr_param.alpha)*0
    elif prior == "simple":
        print("simple prior")
        phiY *= (1-blr_param.alpha)
        phiphiT *= (1-blr_param.alpha)
    elif prior == "sdp":
        print("SDP prior")                                                     #batch size
        phiphiT0, _ = information_transfer(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, 50000      , num_actions, feat_dim)
        phiY *= (1-blr_param.alpha)*0
        phiphiT *= (1-blr_param.alpha)*0

    n = np.zeros(num_actions)
    n_samples = blr_param.batch_size if len(replay_buffer) > blr_param.batch_size else len(replay_buffer)
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(n_samples)
    mini_batch_size = 32*num_actions
    for j in tqdm(range((n_samples // mini_batch_size)+1)):
        # obs_t, action, reward, obs_tp1, done = obses_t[j], actions[j], rewards[j], obses_tp1[j], dones[j]
        start_idx = j*mini_batch_size
        end_idx = (j+1)*mini_batch_size if (j+1)*mini_batch_size < len(replay_buffer) else -1
        obs_t = obses_t[start_idx:end_idx] / 255.
        action = actions[start_idx:end_idx]
        reward = rewards[start_idx:end_idx]
        obs_tp1 = obses_tp1[start_idx:end_idx] / 255.
        done = dones[start_idx:end_idx]

        # obs_feat = dqn_feat(obs_t).reshape((obs_t.shape[0], feat_dim))
        # calculating target
        # y = (reward +(1.-done) * gamma * np.max(target(obs_tp1)))
        for k in range(num_actions):
            if obs_t[action == k].shape[0] < 1:
                continue
            phiphiTk, phiYk = blr_ops(obs_t[action == k], action[action == k], reward[action == k], obs_tp1[action == k], done[action == k])
            phiphiT[k] += phiphiTk
            phiY[k] += phiYk
            # phiphiT[k] += np.dot(obs_feat[action == k].transpose(),obs_feat[action == k])
            # phiY[k] += np.dot(y[action == k], obs_feat[action == k]).reshape((feat_dim,))
            n[k] += obs_t[action == k].shape[0]
    a = [blr_param.a0 for _ in range(num_actions)]
    b = [blr_param.b0 for _ in range(num_actions)]
    print(n, np.sum(n))
    # prior_tensor = np.zeros((num_actions,feat_dim,feat_dim))
    # for i in range(num_actions):
    #     prior_tensor[i] = np.eye(feat_dim)
    # prior_tensor *= 1/blr_param.sigma
    # inv, w_mu = blr_helpers(prior_tensor + phiphit/blr_param.sigma_n,phiy)
    # w_mu *= 1/blr_param.sigma_n
    # w_cov = blr_param.sigma*inv
    for i in range(num_actions):
        inv = np.linalg.inv(phiphiT[i]/blr_param.sigma_n + 1/blr_param.sigma * np.eye(feat_dim))
        if prior == "sdp":
            w_mu[i] = np.array(np.dot(inv,(phiY[i]/blr_param.sigma_n + np.dot(phiphiT0[i], last_layer_weights[:,i]))))
        else:
            w_mu[i] = np.array(np.dot(inv,phiY[i]))/blr_param.sigma_n
        w_cov[i] = blr_param.sigma*inv
    return phiphiT, phiY, w_mu, w_cov, a, b

def BayesRegWithPrior(phiphiT, phiY, w_target, replay_buffer,dqn_feat, target_dqn_feat, target, num_actions, blr_param, w_mu, w_cov, last_layer_weights):
    # dqn_ and target_dqn_ are feature extractors for the dqn and target dqn respectivley
    # in the neural linear settings target are the old features and dqn are the new features
    # last_layer_weights are the target last layer weights
    print("with prior")
    batch_size = blr_param.batch_size
    gamma = blr_param.gamma
    feat_dim = blr_param.feat_dim

    if blr_param.first_time:
        phiphiT0 = [0.01*np.eye(feat_dim) for _ in range(num_actions)]
        cov =  [4*np.eye(feat_dim) for _ in range(num_actions)]
        blr_param.first_time = False
    else:
        phiphiT0, cov = information_transfer(phiphiT, dqn_feat, target_dqn_feat, replay_buffer, batch_size, num_actions,
                                             feat_dim)
    phiY *= 0
    phiphiT = phiphiT0 #phiphiT is actually phiphiT0 + phiphiT
    YY = [0 for _ in range(num_actions)]
    n = [0 for _ in range(num_actions)]
    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(len(replay_buffer))
    for j in range(batch_size):
        obs_t, action, reward, obs_tp1, done = obses_t[j], actions[j], rewards[j], obses_tp1[j], dones[j]
        obs_t = obs_t / 255.
        obs_tp1 = obs_tp1 / 255.
        obs_feat = dqn_feat(obs_t[None]).reshape((feat_dim,1))

        phiphiT[int(action)] += np.dot(obs_feat,obs_feat.transpose())
        y = (reward +(1.-done) * gamma * np.max(target(obs_tp1[None])))
        phiY[int(action)] += (obs_feat*y).reshape((feat_dim,))
        YY[int(action)] += y*y
        n[int(action)] += 1
    a = [blr_param.a0 for _ in range(num_actions)]
    b = [blr_param.b0 for _ in range(num_actions)]
    print(n)
    for i in range(num_actions):
        inv = np.linalg.inv(phiphiT[i])
        w_mu[i] = np.array(np.dot(inv,phiY[i]+np.dot(phiphiT0[i],last_layer_weights[:,i])))
        w_cov[i] = inv
    return phiphiT, phiY, w_mu, w_cov, a, b

def learn_neural_linear(env,
          network,
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=10,#100
          checkpoint_freq=10000,
          checkpoint_path=None,
          learning_starts=999,
          gamma=1.0,
          target_network_update_freq=500,
          prioritized_replay=False,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=None,
          prioritized_replay_eps=1e-6,
          param_noise=False,
          callback=None,
          load_path=None,
          ddqn=False,
          prior="no prior",
          actor="dqn",
          **network_kwargs
                        ):
    #Train a deepq model.

    # Create all the functions necessary to train the model
    checkpoint_path = logger.get_dir()
    sess = get_session()
    set_global_seeds(seed)

    blr_params = BLRParams()
    q_func = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[blr_params.feat_dim],
        dueling=bool(0),
    )
    # q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, feat_dim, feat, feat_target, target, last_layer_weights, blr_ops, blr_helpers = deepq.build_train_neural_linear(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noise=param_noise,
        double_q=ddqn,
        actor=actor
    )
    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                       initial_p=prioritized_replay_beta0,
                                       final_p=1.0)
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))

        # BLR
        # preliminearies
        num_actions = env.action_space.n
        w_mu = np.zeros((num_actions, feat_dim))
        w_sample = np.random.normal(loc=0, scale=0.1, size=(num_actions, feat_dim))
        w_target = np.random.normal(loc=0, scale=0.1, size=(num_actions, feat_dim))
        w_cov = np.zeros((num_actions, feat_dim,feat_dim))
        for a in range(num_actions):
            w_cov[a] = np.eye(feat_dim)

        phiphiT = np.zeros((num_actions,feat_dim,feat_dim))
        phiY = np.zeros((num_actions, feat_dim))

        a0 = 6
        b0 = 6
        a_sig = [a0 for _ in range(num_actions)]
        b_sig = [b0 for _ in range(num_actions)]

        yy = [0 for _ in range(num_actions)]

        blr_update = 0

        for t in tqdm(range(total_timesteps)):
            if callback is not None:
                if callback(locals(), globals()):
                    break
            # if t % 1000 == 0:
            #     print("{}/{}".format(t,total_timesteps))
            # Take action and update exploration to the newest value
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t) + exploration.value(t) / float(env.action_space.n))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True

            action = act(np.array(obs)[None], w_sample[None])
            env_action = action
            reset = False
            new_obs, rew, done, _ = env.step(env_action)

            # clipping like in BDQN
            rew = np.sign(rew)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                episode_rewards.append(0.0)
                reset = True

            # sample new w from posterior
            if t > 0 and t % blr_params.sample_w == 0:
                for i in range(num_actions):
                    if blr_params.no_prior:
                        w_sample[i] = np.random.multivariate_normal(w_mu[i], w_cov[i])
                    else:
                        sigma2_s = b_sig[i] * invgamma.rvs(a_sig[i])
                        w_sample[i] = np.random.multivariate_normal(w_mu[i], sigma2_s*w_cov[i])

            if t > learning_starts and t % train_freq == 0:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = np.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = np.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > learning_starts and t % target_network_update_freq == 0:
                # Update target network periodically.
                # when target network updates we update our posterior belifes
                # and transfering information from the old target
                # to our new target
                blr_update += 1
                if blr_update == 10: #10
                    print("updating posterior parameters")
                    if blr_params.no_prior:
                        phiphiT, phiY, w_mu, w_cov, a_sig, b_sig = BayesRegNoPrior(phiphiT, phiY, w_target, replay_buffer, feat,
                                                                                     feat_target, target, num_actions, blr_params, w_mu, w_cov, sess.run(last_layer_weights),prior, blr_ops, blr_helpers)
                    else:
                        phiphiT, phiY, w_mu, w_cov, a_sig, b_sig = BayesRegWithPrior(phiphiT, phiY, w_target, replay_buffer, feat,
                                                              feat_target, target, num_actions, blr_params, w_mu, w_cov, sess.run(last_layer_weights))
                    blr_update = 0

                print("updateing target, steps {}".format(t))
                update_target()
                w_target = w_mu

            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            mean_10ep_reward = round(np.mean(episode_rewards[-11:-1]), 1)
            num_episodes = len(episode_rewards)
            # if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            if t % 10000 == 0: #1000
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.record_tabular("mean 10 episode reward", mean_10ep_reward)
                logger.dump_tabular()

            if (checkpoint_freq is not None and t > learning_starts and
                    num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            load_variables(model_file)

    return act