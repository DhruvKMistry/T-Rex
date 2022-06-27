# T-Rex
Dino is a famous Google Chrome game that we can play when the internet connection is down. We built an algorithm for a robot that learns by making mistakes. Dino game has two modes. Mode one is to train the model through our gameplay, and then we can save the model. Model 2 is loading the model from the file and the game will start training based on the model.
Q-learning is a reinforcement learning algorithm that learns the value of actions in specific states. The environment is the game itself. We have jumps and idle actions. First, we will train the model to observe the game by sending it images of the game. Second, we use user input as actions (i.e. jump or idle) or a q-learning library to decide actions and execute them. After that, we will set a reward or punishment for the action taken. In the end, it will continuously learn from experience to get the best strategy.


## Libraries
[wayou](https://github.com/wayou/t-rex-runner)

[convnetjs](https://github.com/karpathy/convnetjs)
