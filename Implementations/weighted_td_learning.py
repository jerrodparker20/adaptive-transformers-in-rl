from tqdm.autonotebook import tqdm
import gym
import torch
from dqn import DQN, ReplayBuffer
from torch.optim import Adam
from torch.nn.functional import mse_loss
from plot_functions import plot_timesteps_and_rewards

# In[]:

env = gym.make('CartPole-v1')

# In[]:


def run_current_policy(env, policy, epsilon):
    cur_state = env.reset()
    cur_state = torch.Tensor(cur_state)
    total_step = 0
    total_reward = 0.0
    done = False
    while not done:
        action = policy.select_action(cur_state, epsilon)
        next_state, reward, done, info = env.step(action)
        next_state = torch.Tensor(next_state)
        total_reward += reward
        env.render(mode='rgb_array')
        total_step += 1
        cur_state = next_state
    print("Total timesteps = {}, total reward = {}".format(total_step, total_reward))

# In[]:

gamma = 0.95
epsilon = 0.05
learning_rate = 0.01

avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 1
avg_reward = 0.0
avg_timestep = 0

# initialize policy and replay buffer
policy = DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=12)
replay_buffer = ReplayBuffer()

optimizer = Adam(policy.parameters(), lr=learning_rate)
start_episode = 0

# Play with a random policy and see
# run_current_policy(env.env, policy)

train_episodes = 200
pbar_cp = tqdm(total=train_episodes)

# In[]:


# Update the policy by a TD(0)
def update_policy(cur_states, actions, next_states, rewards, dones):
    # target doesnt change when its terminal, thus multiply with (1-done)
    # target = R(st-1, at-1) + gamma * max(a') Q(st, a')
    targets = rewards + torch.mul(1 - dones, gamma * policy(next_states).max(dim=1).values)
    # expanded_targets are the Q values of all the actions for the current_states sampled
    # from the previous experience. These are the predictions
    expanded_targets = policy(cur_states)[:, actions].squeeze(-1)
    optimizer.zero_grad()
    loss = mse_loss(input=expanded_targets, target=targets.detach())
    loss.backward()
    optimizer.step()
    return loss.item()


# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):
    episode_timestep = 0
    episode_reward = 0.0

    done = False

    cur_state = torch.Tensor(env.reset())

    while not done:
        # select action
        action = policy.select_action(cur_state, epsilon)

        # take action in the environment
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state)

        # add the transition to replay buffer
        replay_buffer.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        sample_transitions = replay_buffer.sample()

        # update the policy using the sampled transitions
        update_policy(**sample_transitions)

        episode_reward += reward
        episode_timestep += 1

        cur_state = next_state

    avg_reward += episode_reward
    avg_timestep += episode_timestep

    if (episode_i + 1) % agg_interval == 0:
        avg_history['episodes'].append(episode_i + 1)
        avg_history['timesteps'].append(avg_timestep / float(agg_interval))
        avg_history['reward'].append(avg_reward / float(agg_interval))
        avg_timestep = 0
        avg_reward = 0.0
    pbar_cp.update()

start_episode = start_episode + train_episodes
plot_timesteps_and_rewards(avg_history)
run_current_policy(env, policy, epsilon)
env.close()
