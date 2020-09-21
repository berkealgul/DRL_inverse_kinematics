import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve
from env import ArmEnvironment

name = "arm_2020"
filename = "plots/"+name+".png"
load = False

env = ArmEnvironment()
agent = Agent(alpha=0.001, beta=0.001, input_shape=env.observation_space,
			n_actions=env.action_space[0], tau=0.005, name=name,
			max_action=env.action_range[1], min_action=env.action_range[0])

best_score = -np.inf
score_history = []
n_episodes = 2000000
n_steps = 200
plot_per_episode = 5

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
		print("episode: ", i+1, "/", n_episodes)
		print("step: ", j+1, "/",n_steps)
		print("")
		print("state: ", obs)
		print("info: ", info)
		print("state_ ", obs_)
		print("score ", score)
		print("##############################################")

		if done:
			break

	score_history.append(score)
	avg_score = np.mean(score_history[-100:])

	if avg_score > best_score:
		best_score = avg_score
		agent.save()

	print("##############################################")
	print("")
	print("Episode: ", i+1, " avg_score: %.1f " % avg_score)
	print("")
	print("##############################################")

	if i+1   % plot_per_episode == 0:
		print("PLOTTING")
		x = [i+1 for i in range(n_episodes)]
		plot_learning_curve(x, score_history, filename)


x = [i+1 for i in range(n_episodes)]
plot_learning_curve(x, score_history, filename)
