import gym

from baselines.common import set_global_seeds, tf_util as U
import logging
import gym

if __name__=='__main__':
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(1)
    env_id='InvertedPendulumAdv-v1'
    env=gym.make(env_id)

    env.seed(1)
    gym.logger.setLevel(logging.WARN)
    ac=env.sample_action()

    for _ in range(10000):
        env.render()
        ac.pro=env.action_space.sample()
        ac.adv=env.adv_action_space.sample()
        env.step(ac)