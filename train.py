from unityagents import UnityEnvironment
import numpy as np
from maddpg import MADDPGAgent
import torch
from collections import deque
import matplotlib.pyplot as plt
import yaml
from torch.utils.tensorboard import SummaryWriter

def train_loop(brain_name, env, agent, n_episodes=2000, max_t=2000, goal=0.51, running_average=100, **params):
    """Train an agent with the environment and using hyper parameters
    
    Params
    ======
        brain_name: brain used in the environment
        env: gym environment to interact
        agent: agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        model_wegihts_file: file to save model weights
        params: hyper parameters
    """

    writer = SummaryWriter()
    agent.setWriter(writer)
    averaged_scores = []
    scores_window = deque(maxlen=running_average)  # last running_average score
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations                  # get the current state
        scores = np.zeros(num_agents)
        for i in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations         # get next state 
            rewards = np.array(env_info.rewards)                         # get reward 
            dones = np.array(env_info.local_done)
            agent.step(states, actions.reshape(num_agents, -1), rewards.reshape(num_agents, -1) , next_states, dones.reshape(num_agents, -1))
            
            scores += np.array(env_info.rewards)
            if np.any(dones):
                break

            states = next_states

        writer.add_scalar('episode length', i, i_episode)
        for i in range(num_agents):
            writer.add_scalar(f'rewards/{i}', scores[i], i_episode)

        score = np.max(scores) # get the max of the sum of each agent
        scores_window.append(score)       # save most recent averaged score
        averaged_scores.append(score)                        # save most recent score
        writer.flush()

        print('\rEpisode {} \tAverage Score: {:.2f}\tMax score: {:.2f}'.format(i_episode, np.mean(scores_window), np.max(scores_window)), end="")
        if i_episode % running_average == 0:
            print('\rEpisode {} \tAverage Score: {:.2f}\tMax score: {:.2f}'.format(i_episode, np.mean(scores_window), np.max(scores_window)))
        if len(scores_window) >= running_average and np.mean(scores_window)>=goal:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}\tMax score: {:.2f}'.format(i_episode, np.mean(scores_window), np.max(scores_window)))

            for i,ag in enumerate(agent.agents):
                torch.save(ag.actor_local.state_dict(), f"actor{i}.weight")
                torch.save(ag.critic_local.state_dict(), f"critic{i}.weight")
            break

    writer.close()
    return averaged_scores    


if __name__ == "__main__":
    # Hyper parameters
    params = yaml.load(open("parameters.yaml"), Loader=yaml.FullLoader)["parameters"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = UnityEnvironment(file_name='./Tennis_Linux_NoVis/Tennis.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]
    num_agents = len(env_info.agents)
    print(f"agents = {num_agents} . Action space: {action_size} observation space: {state_size}")
    agent = MADDPGAgent(state_size, action_size, num_agents, device=device, **params)
    max_t=2000

    scores = train_loop(brain_name, env, agent, n_episodes=5000, max_t=5000, goal=0.51, running_average=100, **params)


    # Plot Statistics (Global scores and averaged scores)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Tennis Environment Average Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()

