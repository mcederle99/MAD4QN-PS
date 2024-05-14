import argparse
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec
import numpy as np
from dqn.dueling_ddqn_agent import DuelingDDQNAgent
import torch
from scenarios import scenarios
from util_rgb import make_env, position2road, roads2t_i

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument('--headless', action='store_true', help='Not visualise the simulation')
args = parser.parse_args()

if __name__ == '__main__':

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    )

    agent_specs = {"Agent-0": agent_spec, "Agent-1": agent_spec, "Agent-2": agent_spec, "Agent-3": agent_spec}
    agent_names = ['Agent-0', 'Agent-1', 'Agent-2', 'Agent-3']

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env("smarts.env:hiway-v0", agent_specs, scenarios, args.headless, args.seed)

    mem_size = 1

    agents_straight = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                       n_actions=2, mem_size=int(mem_size * 1.5), eps_min=0.01,
                                       batch_size=256, replace=1000, eps_dec=1e-6, chkpt_dir='models',
                                       algo='DuelingDDQNAgents',
                                       env_name=f'agent_straight_{args.seed}')
    agents_left = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                   n_actions=2, mem_size=int(mem_size * 1.5), eps_min=0.01,
                                   batch_size=256,
                                   replace=1000, eps_dec=1e-6, chkpt_dir='models', algo='DuelingDDQNAgents',
                                   env_name=f'agent_left_{args.seed}')
    agents_right = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                    n_actions=2, mem_size=mem_size, eps_min=0.01, batch_size=256,
                                    replace=1000, eps_dec=1e-6, chkpt_dir='models', algo='DuelingDDQNAgents',
                                    env_name=f'agent_right_{args.seed}')

    agents_straight.load_models()
    agents_left.load_models()
    agents_right.load_models()

    num_runs = 81
    num_steps = 1000

    collisions = []
    travel_time = []
    waiting_time = []
    avg_speed = []

    action_dict = {0: 'keep_lane', 1: 'slow_down'}

    for i in range(num_runs):
        vel = []
        tts = [0, 0, 0, 0]
        ttl = [0, 0, 0, 0]
        done, crash = False, False

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

        for _ in range(num_steps):
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

            observations, rewards, _, infos = env.step(agent_actions)

            speeds = []
            for idx, k in enumerate(agent_names):
                if k in list(observations.keys()):
                    tts[idx] += infos[k]['env_obs'][0]
                    speed = infos[k]['env_obs'][5].speed
                    speeds.append(speed)
                    if speed < 0.1:
                        ttl[idx] += infos[k]['env_obs'][0]

            if len(list(observations)) == 0:
                done = True
            else:
                vel.append(np.mean(speeds))
                
            if -10 in rewards:
                crash = True

            if crash or done:
                break

        travel_time.append(np.mean(tts))
        avg_speed.append(np.mean(vel))
        waiting_time.append(np.mean(ttl))
        if crash:
            collisions.append(1)
        else:
            collisions.append(0)

    np.save(f'data/travel_time_mad4qn_{args.seed}.npy', travel_time)
    np.save(f'data/waiting_time_mad4qn_{args.seed}.npy', waiting_time)
    np.save(f'data/avg_speed_mad4qn_{args.seed}.npy', avg_speed)
    np.save(f'data/collisions_mad4qn_{args.seed}.npy', collisions)

    print(f'Finished seed {args.seed}')
    env.close()
