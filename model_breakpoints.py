# Often, we want to see what a musculoskeletal model consists of (i.e. get a list of
# the muscles, bodies, their properties, etc.). We can load scone and manually browse
# through it all, but if we just want to do so while we're running and debugging code,
# we need some setup. This file takes just the gym environment, runs it in a single
# process, and lets us add breakpoints where we want them so that we can inspect the
# model

import gym
import env

# create the sconegym env
env = gym.make("H0918-v0") # refers us to env/__init__.py which refers to scone_env.py
# Should be able to add a breakpoint any time after this to inspect self.model
action = env.action_space.sample()
for ep in range(100):
    if ep % 10 == 0:
        env.store_next_episode()  # Store results of every 10th episode

    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = env.action_space.sample()
        # applies action and advances environment by one step
        next_state, reward, done, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward

        # check if done
        if done or (ep_steps >= 1000):
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f}; \
                com={env.model.com_pos()}"
            )
            break

env.close()
