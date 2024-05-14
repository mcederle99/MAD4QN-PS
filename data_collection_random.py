import argparse
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo.agent_spec import AgentSpec
import numpy as np
from scenarios import scenarios
from util_rgb import make_env

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

    np.random.seed(args.seed)

    env = make_env("smarts.env:hiway-v0", agent_specs, scenarios, args.headless, args.seed)

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
        env.set_evaluation_scenario(i)

        observations = env.reset()

        for _ in range(num_steps):
            actions = [0 for _ in range(len(agent_names))]
            agent_actions = {}
            for q, j in enumerate(agent_names):
                if j in list(observations.keys()):
                    actions[q] = np.random.randint(2)
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

    np.save(f'datarandom/travel_time_random_{args.seed}.npy', travel_time)
    np.save(f'datarandom/waiting_time_random_{args.seed}.npy', waiting_time)
    np.save(f'datarandom/avg_speed_random_{args.seed}.npy', avg_speed)
    np.save(f'datarandom/collisions_random_{args.seed}.npy', collisions)

    print(f'Finished seed {args.seed}')
    env.close()
