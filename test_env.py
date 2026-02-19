import gym
import sconegym
import env

env = gym.make("H0918-v0")
action = env.action_space.sample()
for ep in range(10):
    if ep % 10 == 0:
        env.store_next_episode()

    ep_steps = 0
    ep_tot_reward = 0
    state = env.reset()

    while True:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        ep_steps += 1
        ep_tot_reward += reward

        if terminated or truncated or (ep_steps >= 1000):
            print(
                f"Episode {ep} ending; steps={ep_steps}; reward={ep_tot_reward:0.3f}; \
                com={env.model.com_pos()}"
            )
            break

env.close()
