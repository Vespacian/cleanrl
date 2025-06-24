# Reward over Time Plots for td3_continuous from CleanRL
## Eric Bae


# Preliminary Thoughts
the parts i will need
- reward over time plots
- different architectures and action modeling strategies
    - compare to generic log-prob minimization
- include a chart with method name, parameter used, and final reward

steps
- first i think im going to open the files and make sure i can process them
- i want to set up a pipeline that i can modify to try different architecutres and action models
    - for pipeline, just setup basic training on generic log-prob minimization
- then research into these different methods

# Preliminary Findings
Loaded up both pytorch files for data and actor weights
- dict_keys(['mean_reward', 'std_reward', 'observations', 'actions'])
- odict_keys(['action_scale', 'action_bias', 'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias', 'fc_mean.weight', 'fc_mean.bias', 'fc_logstd.weight', 'fc_logstd.bias'])
- noticed 2400000 observations/actions pairs in data
- action_scale is the action space of half cheetah
- not really sure what action_bias is
- 256 fc1/fc2 weight and biases

![from gymnasium website on halfcheetah](img/action_space.png)

# Setting up pipeline
First, I wanted to create the training loop, evaluation function, and plotting function. I decided to start with baseline

For the training loop:
- decided to start with mini-batch gradient descent 
    - not sure if full batch or stochastic would be better so just went with the mini batch
- decided 1000 per batch instead of 1024 to guarantee an even num of elements per split. might adjust this number later
- intuitively, my training loop code is very similar to my checkpoint code

First try:
- in my first initial run, I decided to define a few hyperparameters: `batch_size=1000`, `lr=1e-3`, `eval_freq=10`, and `N=5`
- `eval_freq` is how often I call the eval function in between iterations
- `N` defines how many episodes to run in the eval function
-  my main goal for this first run was to have a proof of concept working that I can work off on. Because of this, all hparams were pretty arbitrary
- I decided to just try a Normal distribution with the log_prob to see what would happen

![results of first try](img/first_try.png)

- some observations I made: with current hparams, there are 240 points on the graph and rewards are ~5000-6000


# Different Methodologies Tried


