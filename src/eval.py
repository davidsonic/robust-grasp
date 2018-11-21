import argparse
import os

import gym
from baselines.common import set_global_seeds, tf_util as U

import MlpPolicy
import PPO_RARL

os.environ['CUDA_VISIBLE_DEVICES']='0'

def policy_fn(name, ob_space, ac_space):
    return MlpPolicy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                               hid_size=64, num_hid_layers=2)


def eval(env_id, seed, save_path, is_adv, force):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    test_env=gym.make(env_id)
    test_env.seed(seed)

    if is_adv:
        test_env.update_adversary(force)

    ob_space=test_env.observation_space
    ac_space=test_env.action_space
    model = policy_fn("pro_pi", ob_space, ac_space)  # Construct network for new policy
    model.load(save_path)
    print('model loaded')
    rew= PPO_RARL.test(model, test_env, is_adv)

    test_env.close()
    return rew


if __name__=='__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # environment must be correct
    parser.add_argument('--env', help='environment ID', default='HopperAdv-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--save_path', default='/tmp/gym/test3/pro_best_model', help='save model path')
    parser.add_argument('--adv', default=False, help='add adv')
    parser.add_argument('--force', default=1, help='adv force')
    args = parser.parse_args()
    print('Testing params')
    print(args)
    reward= eval(args.env, args.seed, args.save_path, args.adv, args.force)

    print(reward)

