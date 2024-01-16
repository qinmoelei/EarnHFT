# Training Preliminary
## Data 
The data contains 2 kinds of information: Reward related features and state related features.
### Reward Related Features 
The data should consist of basic 5 level LOB features `ask1_price`,`ask1_size`,`bid1_price`,`bid1_size`...
to help calcualte the order-taker style of conducted price, liquidity in the market and our avaliable action.
### State Related Features
Since our method is hierarchical reinforcement learning, the state related features also contains 2 levels: second level & mintue level. The second level features are updated every second and the mintue level features are updated every mintue.
## Data Preprocessing
First run [`data/split.sh`](https://github.com/pythagoras-investment/strategy-ai/blob/main/data/split.sh) to split the dataset into train, valid and test dataset and break down the training dataset into little chunks, where 
- `data_path` stands for the folder under which the original data is stored.
- `chunk_length` stands for the chunk length of the training dataset. During training, for the considertation of the RAM, we split the whole training dataset into little chunks. This movement also help to solve the problem of forgetting and gain a better generalization ability.
- `future_sight`, the extra length in the little chunk to help us to build a longer sight q table.
Then run `tool/label.sh` to segement the valid dataset for refine strategy pool.



# Training Process
## Low Level RL Training
Run the script in [`script/{your_dataset}/low_level/train.sh`](https://github.com/pythagoras-investment/strategy-ai/blob/main/script/BTCUSDT/low_level/train.sh). 
Here are where the coffients stand for
- `beta` stands for the sampling preference coffient, higher `beta` indicates more extreme preference over good market dynamics.
- `train_data_path` where the training data chunks are stored.
- `dataset_name` indicating the name of the dataset, help to restore the trained model.
- `max_holding_number` indicating the max number of the coin we can hold in our hand. You can view it as initial money in another form. Based on my experience, the current method works the best when the price of max number of the coin is around 300. It means that if you are using BTC as your training choice, this coffient should be around 0.01.
- `transcation_cost` indicates the commission fee rate.

## Comprehensive Evaluation of the Low Level Agent
Run [`script/{your_dataset}/low_level/valid_multi.sh`](https://github.com/pythagoras-investment/strategy-ai/blob/main/script/BTCUSDT/low_level/valid_multi.sh).
The result will be automatically stored in the path of trained model. Please notice that this might take up many nodes in the server. So make sure that there are enough nodes in the server by using the command `df -i`
## Refine Strategy 
Run [`script/{your_dataset}/high_level/pick.sh`](https://github.com/pythagoras-investment/strategy-ai/blob/main/script/BTCUSDT/high_level/pick.sh)
This script will search all the result and form the final strategy pool under the folder `result_risk/{your_dataset}/potential_model`
## High Level RL Training
Run the script in [`script/{your_dataset}/high_level/train.sh`](https://github.com/pythagoras-investment/strategy-ai/blob/main/script/BTCUSDT/high_level/train.sh). 
## High Level RL Testing
Run the script in [`script/{your_dataset}/high_level/test.sh`](https://github.com/pythagoras-investment/strategy-ai/blob/main/script/BTCUSDT/high_level/test.sh)




