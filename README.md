# Deep-Fighter-Reinforcement-Learning-Project
Reinforcement Learning Undergraduate Honors thesis project. Using the EmuHawk Emulator and the python TensorForce library, a RL agent playing as Ryu was trained against Street Fighter II Turbo's native Zangief CPU at varying levels of difficulty.

# Setup:
Training was performed on an Ubuntu system using two Quadro P620 gpus with a GeForce GTX 1080 used in the last two months of the project. The 1080 almost doubled training speed and allowed for larger network architecture size.
The agent running on the Ubuntu system's gpu trained by communicating with a Windows device through the local network that could run the Emulation software EmuHawk. 
EmuHawk's source code was altered to follow a training loop:
1. Receive action to perform from agent
2. Advance 4 frames
3. Send RAM values and screenshot of game to agent
4. Repeat

# Results
The Results folder contains match results of two trained agents. Due to the deterministic nature of the game's CPU and the trained agent, there were only a discrete number of possible outcomes for each level of CPU. Saliency was used as a visualization tool to tell what the agent payed attention to in its network's output.
