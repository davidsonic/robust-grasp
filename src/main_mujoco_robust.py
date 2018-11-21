"""
Training script for RARL framework
"""

import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
import logging

import PPO_RARL
import gym
from baselines.common import tf_util as U
from baselines.results_plotter import load_results
from baselines.bench import Monitor
import numpy as np
import MlpPolicy
from baselines.common import set_global_seeds

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)

log_dir='/tmp/gym/rarl_2'
os.makedirs(log_dir, exist_ok=True)
best_mean_reward, n_steps = -np.inf, 0

def plot_callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 2 == 0:
        # Evaluate policy performance
        # x, y = ts2xy(load_results(log_dir), 'timesteps')
        df=load_results(log_dir)
        x=np.cumsum(df.l.values)
        y=df.r.values
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['pro_pi'].save(log_dir + '/pro_best_model')
                _locals['adv_pi'].save(log_dir+'/adv_best_model')
    n_steps += 1
    return False



def make_env(env_id, rank, seed):
    def _init():
        env=gym.make(env_id)
        env.seed(seed+1)
        return env
    set_global_seeds(seed)
    return _init


def train(env_id, num_iters, seed, n=1, success_reward=1000, save_path='model/new_model'):

    U.make_session(num_cpu=4).__enter__()
    set_global_seeds(seed)

    env = gym.make(env_id)
    env.update_adversary(n)
    env=Monitor(env, log_dir, allow_early_resets=True)
    env.seed(seed)

    test_env = gym.make(env_id)
    test_env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    # debug not working
    # num_cpu=4
    # env=SubprocVecEnv([make_env(env_id,i,seed) for i in range(num_cpu)])

    rew = PPO_RARL.learn(env, test_env, policy_fn,
                         timesteps_per_batch=2048,
                         clip_param=0.2, entcoeff=0.0,
                         optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
                         gamma=0.99, lam=0.95, schedule='constant', success_reward=success_reward,
                         save_path=save_path, max_iters=num_iters, callback=plot_callback
                         )

    env.close()

    return rew


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='HopperAdv-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=10)
    parser.add_argument('--sr', default=3800.0, help='success reward')
    args = parser.parse_args()
    print('Training params')
    print(args)
    model = train(args.env, num_iters=500, seed=args.seed, success_reward=args.sr, save_path=log_dir)
    print(model)
