from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce.execution.base_runner import BaseRunner


import time
from six.moves import xrange
import warnings
from inspect import getargspec
from tqdm import tqdm
from saliency import *
import cv2
from PIL import Image

class SFRunner(BaseRunner):
    """
    Simple runner for non-realtime single-process execution.
    """

    def __init__(self, agent, environment, repeat_actions=1, history=None, id_=0, logger=None, report=100, save_interval=100, save='../Current/Model', rewards=None, plan=None):
        """
        Initialize a single Runner object (one Agent/one Environment).
        Args:
            id_ (int): The ID of this Runner (for distributed TF runs).
        """
        super(SFRunner, self).__init__(agent, environment, repeat_actions, history)

        self.id = id_  # the worker's ID in a distributed run (default=0)
        self.current_timestep = None  # the time step in the current episode
        self.logger = logger
	self.report = report
	self.save_interval = save_interval
	self.save = save
	self.rewards = rewards
	
	if plan is not None:
	    self.plan = iter(plan)
	    self.match = self.plan.next()
	    self.end = self.match["End"]
	    self.environment.reset(match=self.match["Match"])
	else:
	    self.plan = None
	    self.end = 0


    def close(self):
        self.agent.close()
        self.environment.close()

    # TODO: make average reward another possible criteria for runner-termination
    def run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None, deterministic=False,
            summary_report=None, summary_interval=None, timesteps=None, episodes=None, testing=False, sleep=None
            ):
        """
        Args:
            timesteps (int): Deprecated; see num_timesteps.
            episodes (int): Deprecated; see num_episodes.
        """

        # deprecation warnings
        if timesteps is not None:
            num_timesteps = timesteps
            warnings.warn("WARNING: `timesteps` parameter is deprecated, use `num_timesteps` instead.",
                          category=DeprecationWarning)
        if episodes is not None:
            num_episodes = episodes
            warnings.warn("WARNING: `episodes` parameter is deprecated, use `num_episodes` instead.",
                          category=DeprecationWarning)


        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()

        self.agent.reset()
	if testing:
	    num_episodes += self.agent.episode
##        if num_episodes is not None:
  ##          num_episodes += self.agent.episode

    ##    if num_timesteps is not None:
      ##      num_timesteps += self.agent.timestep

        training_start_time = time.time()
	training_start_step = self.agent.timestep
        # add progress bar
        with tqdm(total=num_episodes) as pbar:
	    if not testing:
	        pbar.update(self.agent.episode)

            # episode loop
            while True:


                self.agent.reset()

                # Update global counters.
                self.global_episode = self.agent.episode  # global value (across all agents)
                self.global_timestep = self.agent.timestep  # global value (across all agents)
		episode_start_time = time.time()
                episode_reward = 0
                self.current_timestep = 0

		if self.plan is not None and self.global_episode > self.end and self.end:
		    while True: 
		    	self.match = self.plan.next()
	    	    	self.end = self.match["End"]
			if self.end > self.global_episode:
			    break
	    	    state = self.environment.reset(match=self.match["Match"])
		else:
	            state = self.environment.reset()

                # time step (within episode) loop
                while True:
                    action, logits = self.agent.act(states=state, deterministic=deterministic)
		    ##print(extra)
                    reward = 0
                    for _ in xrange(self.repeat_actions):
                        state, terminal, step_reward = self.environment.execute(action=action, logits=logits)
                        reward += step_reward
                        if terminal:
                            break


                    if max_episode_timesteps is not None and self.current_timestep >= max_episode_timesteps:
                        terminal = True

                    if not testing:
                        self.agent.observe(terminal=terminal, reward=reward)

                    self.global_timestep += 1
                    self.current_timestep += 1
                    episode_reward += reward

                    if terminal or self.agent.should_stop():  # TODO: should_stop also terminate?
                        break

                    if sleep is not None:
                        time.sleep(sleep)

                # Update our episode stats.
                time_passed = time.time() - episode_start_time
		print('Timesteps/sec:' + str((self.global_timestep-training_start_step)/float(time.time()-training_start_time)))
                self.episode_rewards.append(episode_reward)
                self.episode_timesteps.append(self.current_timestep)
                self.episode_times.append(time_passed)

                self.global_episode += 1
                pbar.update(1)

                # Check, whether we should stop this run.
                self.episode_finished()
                if (num_episodes is not None and self.global_episode >= num_episodes) or \
                        (num_timesteps is not None and self.global_timestep >= int(num_timesteps)) or \
                        self.agent.should_stop() and (not testing):
                    break

    # TODO: make average reward another possible criteria for runner-termination
    def saliency_run(self, num_timesteps=None, num_episodes=None, max_episode_timesteps=None, deterministic=False,
            summary_report=None, summary_interval=None, timesteps=None, episodes=None, testing=False, sleep=None,
            length=1, path='../Current/Saliency.avi'):
        """
        Args:
            timesteps (int): Deprecated; see num_timesteps.
            episodes (int): Deprecated; see num_episodes.
        """
        # deprecation warnings
        if timesteps is not None:
            num_timesteps = timesteps

        if episodes is not None:
            num_episodes = episodes

        # Keep track of episode reward and episode length for statistics.
        self.start_time = time.time()
	print(self.agent.states['state']['shape'])
        [y, x, d] = self.agent.states['state']['shape']
        ep = 0
	#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vid = cv2.VideoWriter(path, 0, 8, (x, y))
	og = cv2.VideoWriter('./Normal.avi', 0, 8, (x, y))
        
        self.agent.reset()
	if testing:
	    num_episodes += length
	num = 0
        with tqdm(total=num_episodes) as pbar:
	    if not testing:
	        pbar.update(self.agent.episode)
            # episode loop
            while True:
                episode_start_time = time.time()

                self.agent.reset()

                # Update global counters.
                self.global_episode = self.agent.episode  # global value (across all agents)
                self.global_timestep = self.agent.timestep  # global value (across all agents)
		start = self.agent.timestep
                episode_reward = 0
                self.current_timestep = 0

		if self.plan is not None and self.global_episode > self.end and self.end:
		    while True: 
		    	self.match = self.plan.next()
	    	    	self.end = self.match["End"]
			if self.end > self.global_episode:
			    break
	    	    state = self.environment.reset(match=self.match["Match"])
		else:
	            state = self.environment.reset()

                # time step (within episode) loop
                while True:
                    action, logits = self.agent.act(states=state, deterministic=deterministic)

                    scores = score_frame(self.agent, logits, state)

		    frame = state*255
		    og.write(frame[:,:,::-1].astype('uint8'))
		    for i in range(0,len(logits)):
                    	frame = salienize(scores[:,:,i], frame, channel=i+1, sigma=1, normVal=128)
		    


		    img = Image.fromarray(frame, "RGB")
	    	    img.save("./Saliency/frame" + str(num) + ".png")
		    num += 1
		    cvFrame = frame[:,:,::-1]

                    vid.write(cvFrame)
                    
                    reward = 0
                    state, terminal, reward = self.environment.execute(action=action, logits=logits)

                    self.global_timestep += 1
                    self.current_timestep += 1
                    episode_reward += reward

                    if terminal or self.agent.should_stop():  # TODO: should_stop also terminate?
                        break
		    
                
                # Update our episode stats.
                time_passed = time.time() - episode_start_time
                self.episode_rewards.append(episode_reward)
                self.episode_timesteps.append(self.current_timestep)
                self.episode_times.append(time_passed)

                self.global_episode += 1
                pbar.update(1)

                ep += 1
                if ep == length:     
                    cv2.destroyAllWindows()
		    vid.release()
		    og.release()	
                    break
                
                if (num_episodes is not None and self.global_episode >= num_episodes) or \
                        (num_timesteps is not None and self.global_timestep >= int(num_timesteps)) or \
                        self.agent.should_stop() and (not testing):
                    break

    def episode_finished(self):
        print('Episode ' + str(self.episode) + ' reward: ' + str(self.episode_rewards[-1]))
        if self.logger is not None:
            if self.episode % self.report == 0:		
                steps_per_second = self.timestep / (time.time() - self.start_time)
                self.logger.info("Finished episode {:d} after {:d} timesteps. Steps Per Second {:0.2f}".format(
                    self.agent.episode, self.episode_timestep, steps_per_second
                ))
                self.logger.info("Average of last 500 rewards: {:0.2f}".
                            format(sum(self.episode_rewards[-500:]) / min(500, len(self.episode_rewards))))
                self.logger.info("Average of last 100 rewards: {:0.2f}".
                            format(sum(self.episode_rewards[-100:]) / min(100, len(self.episode_rewards))))
	if not self.episode % self.save_interval:                
            self.logger.info("Saving agent.")
            self.agent.save_model(self.save)
	self.rewards.write("%d," %self.episode_rewards[-1])

        return True

    # keep backwards compatibility
    @property
    def episode_timestep(self):
        return self.current_timestep

