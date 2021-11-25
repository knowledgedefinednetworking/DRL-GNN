# Instructions to execute

1. First, create the virtual environment and activate the environment.
```ruby
virtualenv -p python3 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```ruby
pip install -r requirements.txt
```

3. Register custom gym environment.
```ruby
pip install -e gym-environments/
```

4. Now we are ready to train a DQN agent. To do this, we must execute the following command. Notice that inside the *train_DQN.py* there are different hyperparameters that you can configure to set the training for different topologies, to define the size of the GNN model, etc.
```ruby
python train_DQN.py
```

5. Now that the training process is executing, we can see the DQN agent performance evolution by parsing the log files.
```ruby
python parse.py -d ./Logs/expsample_DQN_agentLogs.txt
```

6. Finally, we can evaluate our trained model on different topologies executing the command below. Notice that in the *evaluate_DQN.py* script you must modify the hyperparameters of the model to match the ones from the trained model.
```ruby
python evaluate_DQN.py -d ./Logs/expsample_DQN_agentLogs.txt
```