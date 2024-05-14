import collections
import gym
import numpy as np
import cv2

roads2t_i = {'EW': 'straight', 'ES': 'left', 'EN': 'right',
             'WE': 'straight', 'WN': 'left', 'WS': 'right',
             'SN': 'straight', 'SW': 'left', 'SE': 'right',
             'NS': 'straight', 'NE': 'left', 'NW': 'right'}


def eval_policy(agents_straight, agents_left, agents_right, env, eval_episodes=81):
    returns_list = []
    agent_names = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
    action_dict = {0: 'keep_lane', 1: 'slow_down'}

    for i in range(eval_episodes):
        score = 0
        turning_intentions = {}
        env.set_evaluation_scenario(i)
        _ = env.reset()
        observations, _, _, infos = env.step({'Agent-0': 'slow_down', 'Agent-1': 'slow_down',
                                              'Agent-2': 'slow_down', 'Agent-3': 'slow_down'})
        for k in agent_names:
            x1 = infos[k]['env_obs'][5].mission.start.position.x
            y1 = infos[k]['env_obs'][5].mission.start.position.y
            start = position2road([x1, y1])

            x2 = infos[k]['env_obs'][5].mission.goal.position.x
            y2 = infos[k]['env_obs'][5].mission.goal.position.y
            goal = position2road([x2, y2])

            turning_intentions[k] = roads2t_i[start + goal]

        episode_ended = False
        ep_steps = 0

        while not episode_ended and ep_steps < 1000:
            actions = [0 for _ in range(len(agent_names))]
            agent_actions = {}
            for q, j in enumerate(agent_names):
                if j in list(observations.keys()):
                    if turning_intentions[j] == 'straight':
                        actions[q] = agents_straight.choose_action(observations[j], evaluate=True)
                    elif turning_intentions[j] == 'left':
                        actions[q] = agents_left.choose_action(observations[j], evaluate=True)
                    elif turning_intentions[j] == 'right':
                        actions[q] = agents_right.choose_action(observations[j], evaluate=True)
                    agent_actions[j] = action_dict[actions[q]]

            observations, rewards, _, _ = env.step(agent_actions)
            score += sum(rewards)

            if -10 in rewards or len(list(observations)) == 0:
                episode_ended = True

            ep_steps += 1

        returns_list.append(score)

    env.modify_probs(returns_list)
    avg_reward = sum(returns_list) / eval_episodes
    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    env.set_evaluation_scenario(-1)
    return avg_reward, returns_list


def position2road(position):
    if position[0] > 110:
        return 'E'
    elif position[0] < 80:
        return 'W'
    elif position[1] < 90:
        return 'S'
    else:
        return 'N'


class Reward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        wrapped_reward = self._reward(obs, reward)

        return obs, wrapped_reward, done, info

    def _reward(self, obs, env_reward):
        num_vehs = len(obs.keys())
        reward = [0 for _ in range(num_vehs)]
        agent_names = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
        w = 0
        for i, j in enumerate(agent_names):
            if j in obs.keys():

                if obs[j][4][5]:
                    reward[w] -= 1

                elif len(obs[j][4][0]) != 0:
                    reward[w] -= 10

                elif obs[j][4][6]:
                    reward[w] += 10

                else:
                    reward[w] += env_reward[j]

                w += 1

        return np.float64(reward)


class Observation(gym.ObservationWrapper):
    def __init__(self, shape, env: gym.Env):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=self.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        agent_names = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
        new_obs = {}
        for i in agent_names:
            if i in obs.keys():
                new_obs[i] = obs[i][13].data
                new_obs[i] = cv2.resize(new_obs[i], self.shape[1:], interpolation=cv2.INTER_AREA)
                new_obs[i] = np.array(new_obs[i], dtype=np.uint8).reshape(self.shape)
                new_obs[i] = new_obs[i] / 255.0
        return new_obs


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),
                                                env.observation_space.high.repeat(repeat, axis=0),
                                                dtype=np.float32)
        self.stack = [collections.deque(maxlen=repeat), collections.deque(maxlen=repeat),
                      collections.deque(maxlen=repeat), collections.deque(maxlen=repeat)]
    
    def reset(self):
        self.stack[0].clear()
        self.stack[1].clear()
        self.stack[2].clear()
        self.stack[3].clear()
        
        observation = self.env.reset()
        agent_names = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
        for i, j in enumerate(agent_names):
            for _ in range(self.stack[i].maxlen):
                self.stack[i].append(observation[j])
        obs_dict = {}
        obs_dict['Agent-0'] = np.array(self.stack[0]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-1'] = np.array(self.stack[1]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-2'] = np.array(self.stack[2]).reshape(self.observation_space.low.shape)
        obs_dict['Agent-3'] = np.array(self.stack[3]).reshape(self.observation_space.low.shape)
        
        return obs_dict

    def observation(self, observation):
        agent_names = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']
        obs_dict = {}

        for i, j in enumerate(agent_names):
            if j in observation.keys():
                self.stack[i].append(observation[j])
                obs_dict[j] = np.array(self.stack[i]).reshape(self.observation_space.low.shape)

        return obs_dict


def make_env(env_name, agent_specs, scenario_path, headless, seed) -> gym.Env:
    # Create environment
    env = gym.make(
        env_name,
        scenarios=scenario_path,
        agent_specs=agent_specs,
        headless=headless,  # If False, enables Envision display.
        visdom=False,  # If True, enables Visdom display.
        seed=seed,
        
    )

    env = Reward(env=env)
    env = Observation(shape=(48, 48, 3), env=env)
    env = StackFrames(env, repeat=3)

    return env
