import numpy as np

import logging
import json
import ast
import argparse
import os
import time
import sys
import subprocess
import pdb


from tensorforce.agents import Agent
from tensorforce.execution import Runner
from SFEnvironmentV2 import SFIIEnvironment
from SFRunner import SFRunner




def main():
    parse = argparse.ArgumentParser()

    #Arguments that can be parsed to change training format
    parse.add_argument('-n', '--network', default= '../Config/dqnNetDef.txt',
                       help='Specifies network layer architecture')
    parse.add_argument('-s', '--save', default='../Current/Model',
                       help="File to save agent's model to")
    parse.add_argument('-a', '--agent', default='../Config/dqnAgentDef.txt',
                       help="Agent configuration file")
    parse.add_argument('-e', '--episodes', type=int, default=30000,
                       help='Number of episodes to train for')
    parse.add_argument('-f', '--file', default='Zangief3Star',
                       help='Default save file to load match from')
    parse.add_argument('-r', '--results', default='../Current/Results.txt',
                       help='File to save training results to')
    parse.add_argument('-sm', '--summary', default='../Current/Summary', 
		       help='Folder to save summary to')
    parse.add_argument('-t', '--timesteps', default=None,
		       help='Timesteps to train for')
    parse.add_argument('-T', '--test', default=True,
                       help="True if training is not performed")
    parse.add_argument('-rp', '--report', default=100,
		       help='How many episodes between reporting data')
    parse.add_argument('-S', '--saliency', default=2,
                       help='Number of episodes to do saliency testing for')
    parse.add_argument('-se', '--save_episodes', default=100,
			help='Episodes between saving Agent')
    parse.add_argument('-l', '--load', help ="Load agent from this dir")
    parse.add_argument('-p', '--plan', default='../Config/plan.txt', help='Training plan')
    parse.add_argument('-v', '--video', default='./Saliency.avi',
                       help='Path to save saliency video to')


    #Check arguments parsed
    argIn = parse.parse_args()

   
    #Open agent specification and pass in arguments
    with open(argIn.agent, 'r') as fp:
        agentSpec = json.load(fp)
        agentSpec["saver"]["directory"] = argIn.save
        agentSpec["summarizer"]["directory"] = argIn.summary


    #Open network specification
    with open(argIn.network, 'r') as fp:
        networkSpec = json.load(fp)

    #Ensure save directory exists
    if argIn.save:
        save_dir = os.path.dirname(argIn.save)
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir, 0o755)
            except OSError:
                raise OSError("Cannot save agent to dir {} ()".format(save_dir))

    if argIn.plan:
	plan = json.load(open(argIn.plan, 'r'))
	plan = ast.literal_eval(json.dumps(plan))
	if not os.path.isfile("../Current/plan.txt"):
	    with open(argIn.plan, 'r') as fp:
                with open("../Current/plan.txt", "w") as f1:
                    for line in fp:
                        f1.write(line)
    else:
	plan = None   

    if argIn.test:
	argIn.results = '../Current/TestingResults.txt'

    # Create an SFIITurbo environment
    env = SFIIEnvironment(argIn.results, defaultSave=argIn.file)

    #Run emulator and connect
    env.connect()

    states = env.states()
    actions = env.actions()

    #Create DQN Agent from spec
    agent = Agent.from_spec(spec=agentSpec,
                            kwargs=dict(
                                states=env.states(),
                                actions=env.actions(),
                                network=networkSpec
                                )
                            )
    
    if argIn.load:
        load_dir = os.path.dirname(argIn.load)
        if not os.path.isdir(load_dir):
            raise OSError("Could not load agent from {}: No such directory.".format(load_dir))
        agent.restore_model(argIn.load)


    initialState = env.reset()

    if argIn.test:
        rewardSum = open('../Current/TestingRewards.txt', 'a')
    else:
	rewardSum = open('../Current/RewardResults.txt', 'a')

    #Copy Agent and network definitions
    if not os.path.isfile("../Current/dqnAgentDef.txt"):
        with open(argIn.agent, 'r') as fp:
            with open("../Current/dqnAgentDef.txt", "w") as f1:
                for line in fp:
                    f1.write(line)
        with open(argIn.network, 'r') as fp:
            with open("../Current/dqnNetDef.txt", "w") as f1:
                for line in fp:
		    f1.write(line)
    #Create runner
    runner = SFRunner(agent, env, report=argIn.report, 
	plan=plan, save_interval=argIn.save_episodes, save=argIn.save, rewards=rewardSum)

    #Begin training
    runner.saliency_run(num_episodes= argIn.episodes, testing=argIn.test, length=int(argIn.saliency), path=argIn.video)

    runner.close()
    rewardSum.close()
    print("Video capture complete")

if __name__ == '__main__':
    main()
