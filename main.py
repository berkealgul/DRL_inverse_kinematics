import os
import numpy as np
from datetime import datetime
from agent import Agent
from utils import plot_learning_curve
from env import ArmEnvironment

name = "arm_2020"
filename = "plots/"+name+".png"
logname = "logs/"+name+".txt"
load = False

env = ArmEnvironment()
agent = Agent(alpha=0.001, beta=0.001, input_shape=env.observation_space,
			n_actions=env.action_space[0], tau=0.005, name=name,
			max_action=env.action_range[1], min_action=env.action_range[0])

best_score = -np.inf
score_history = []
n_episodes = 2000000
n_steps = 200
logging_per_episode = 5

# Logging for beginnig
with open(logname, 'w') as f:
	now = datetime.now()
	date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
	f.write("[LOG] Training begins" + os.linesep)
	f.write("[LOG] Training Device: "+ str(agent.actor.device) + os.linesep)
	f.write("[LOG] Time: "+ date_time+ os.linesep)
	f.write(os.linesep)


if load:
	agent.warmup = 0
	try:
		agent.load()
	except:
		print("Error while loading...")

for i in range(n_episodes):
	obs = env.reset()
	done = False
	score = 0
	step = 0

	for j in range(n_steps):
		action = agent.choose_action(obs)
		obs_, reward, done, info = env.step(action)
		agent.store_memory(obs, obs_, action, reward, done)
		agent.learn()
		obs = obs_
		score += reward

		print("##############################################")
		print("Episode: ", i+1, "/", n_episodes)
		print("Step: ", j+1, "/",n_steps)
		print("")
		print("State: ", obs)
		print("Info: ", info)
		print("State_: ", obs_)
		print("Score: ", score)
		print("Training device: ", agent.actor.device)
		print("##############################################")

		if done:
			break

	score_history.append(score)
	avg_score = np.mean(score_history[-100:])

	if avg_score > best_score:
		best_score = avg_score
		agent.save()

	# Logging for timestamps
	if (i+1) % logging_per_episode == 0:
		print("LOGGING")
		x = [a+1 for a in range(i+1)]
		plot_learning_curve(x, score_history, filename)

		with open(logname, 'a') as f:
			now = datetime.now()
			date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
			f.write("[LOG]" +" Episode: "+ str(i+1)+"/"+str(n_episodes)+
					" avg_score: %.1f " % avg_score+" Time: "+ date_time + os.linesep)


x = [i+1 for i in range(n_episodes)]
plot_learning_curve(x, score_history, filename)
