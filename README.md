# Interactive DRL
Interactive environment for training and testing a simple Deep Reinforcement Learning agent based on &epsilon;-Greedy Q-Learning.

![image](https://github.com/MattiaFerrarini/Interactive-DRL/assets/119322415/c2eff230-be90-4cb9-8869-a0453df9fed1)


## Requirements
```bash
Python==3.12.1
numpy==1.26.3
torch==2.2.1
pygame==2.5.2
matplotlib==3.8.3
pandas==2.2.1
```

## Training
To train the agent in random environments run
```bash
python training.py
```
and input the desired grid size when prompted. <br>
Once the training is over, the parameters of the model will be saved in `trained_agent.pt`.

## Testing
Once you have trained an agent, you can test it by running
```bash
python testing.py
```
You need to input the desired grid size, which does not need to be the same as the one used during training.

### Creating a custom environment
When you run `testing.py`, you will be able to create a custom environment to test your agent in. 
- Left-click on a grid cell to place/remove an obstacle;
- Enter to start the test.

## Notes
The agent trained is very simple and will probably not perform well in certain environments. <br>
The goal of the project is to simply experiment with DRL.
