import argparse
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec
import numpy as np
from dqn.dueling_ddqn_agent import DuelingDDQNAgent
import torch
from scenarios import scenarios
from util_rgb import make_env, position2road, roads2t_i, eval_policy

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)
parser.add_argument('--headless', action='store_true', help='Not visualise the simulation')
parser.add_argument('--load_checkpoint', action='store_true', help='Load saved models')
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

    if args.load_checkpoint:
        mem_size = 1
    else:
        mem_size = 100000

    agents_straight = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                       n_actions=2, mem_size=int(mem_size*1.5), eps_min=0.01,
                                       batch_size=256, replace=1000, eps_dec=1e-6, chkpt_dir='models',
                                       algo='DuelingDDQNAgents',
                                       env_name=f'agent_straight_{args.seed}')
    agents_left = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                   n_actions=2, mem_size=int(mem_size*1.5), eps_min=0.01, batch_size=256,
                                   replace=1000, eps_dec=1e-6, chkpt_dir='models', algo='DuelingDDQNAgents',
                                   env_name=f'agent_left_{args.seed}')
    agents_right = DuelingDDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, input_dims=(env.observation_space.shape),
                                    n_actions=2, mem_size=mem_size, eps_min=0.01, batch_size=256,
                                    replace=1000, eps_dec=1e-6, chkpt_dir='models', algo='DuelingDDQNAgents',
                                    env_name=f'agent_right_{args.seed}')

    best_score = -1000.0
    total_steps = 1e6

    if args.load_checkpoint:
        agents_straight.load_models()
        agents_left.load_models()
        agents_right.load_models()

    n_steps = 0
    n_episodes = 0
    time_to_eval = False
    scores_list = []
    scores_per_scenario_list = []

    action_dict = {0: 'keep_lane', 1: 'slow_down'}

    if args.load_checkpoint:
        scores, scores_per_scenario = eval_policy(agents_straight, agents_left, agents_right, env)
        scores_list.append(scores)
        scores_per_scenario_list.append(scores_per_scenario)
        raise KeyboardInterrupt

    while n_steps < total_steps:
        turning_intentions = {}
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

            turning_intentions[k] = roads2t_i[start+goal]

        score = 0
        episode_ended = False
        ep_steps = 0

        while not episode_ended and ep_steps < 1000:
            actions = [0 for _ in range(len(agent_names))]
            agent_actions = {}
            for q, j in enumerate(agent_names):
                if j in list(observations.keys()):
                    if turning_intentions[j] == 'straight':
                        actions[q] = agents_straight.choose_action(observations[j])
                    elif turning_intentions[j] == 'left':
                        actions[q] = agents_left.choose_action(observations[j])
                    elif turning_intentions[j] == 'right':
                        actions[q] = agents_right.choose_action(observations[j])
                    agent_actions[j] = action_dict[actions[q]]

            observations_, rewards, dones, _ = env.step(agent_actions)
            score += sum(rewards)
            
            if -10 in rewards or len(list(observations_)) == 0:
                episode_ended = True
            
            w = 0
            for q, j in enumerate(agent_names):
                if j in list(observations.keys()):
                    if j in list(observations_.keys()):
                        done = dones[j]
                        if turning_intentions[j] == 'straight':
                            agents_straight.store_transition(observations[j], actions[q], rewards[w],
                                                             observations_[j], done)
                        elif turning_intentions[j] == 'left':
                            agents_left.store_transition(observations[j], actions[q], rewards[w],
                                                         observations_[j], done)
                        elif turning_intentions[j] == 'right':
                            agents_right.store_transition(observations[j], actions[q], rewards[w],
                                                          observations_[j], done)
                        w += 1
            agents_straight.learn()
            agents_left.learn()
            agents_right.learn()

            observations = observations_
            n_steps += 1
            if n_steps % 5000 == 0:
                time_to_eval = True
            ep_steps += 1

        if time_to_eval:
            scores, scores_per_scenario = eval_policy(agents_straight, agents_left, agents_right, env)
            scores_list.append(scores)
            scores_per_scenario_list.append(scores_per_scenario)
            if scores_list[-1] > best_score:
                agents_straight.save_models()
                agents_left.save_models()
                agents_right.save_models()
                best_score = scores_list[-1]
            time_to_eval = False

            np.save(f'results/mad4qn_{args.seed}.npy',
                    np.array(scores_list))
            np.save(f'results/mad4qn_{args.seed}_per_scenario.npy',
                    np.array(scores_per_scenario_list))

        n_episodes += 1

        if len(scores_list) > 0:
            print('episode ', n_episodes, 'score: ', score,
                  'last ev score %1.f best ev score %1.f epsilon %.2f' % (
                  scores_list[-1], best_score, agents_left.epsilon),
                  'steps ', n_steps)
        else:
            print('episode ', n_episodes, 'score: ', score,
                  'last ev score %1.f best ev score %1.f epsilon %.2f' % (
                  best_score, best_score, agents_left.epsilon),
                  'steps ', n_steps)
    env.close()
