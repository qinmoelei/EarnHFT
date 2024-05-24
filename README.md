The offical implementation of the AAAI 2024 [EarnHFT:Efficient hierarchical reinforcement learning for high frequency trading](https://arxiv.org/pdf/2309.12891.pdf).

# Data
For data preprocessing part, please refer to data_preprocess/README.md.

We download the data from [tardis](https://tardis.dev/). You might need to purchase a API key to fully utilize our code.

We first download the data from tardis, then do some preprocess to use the dataframe to construct the corresponding RL environment in the Algorithm part.

# Algorithm
For algorithm part, please refer to EarnHFT_Algorithm/README.md

We first train the low level agents which operates on a second-level with different preference parameter `beta`.

We then evaluate the low-level agents with valid data which is divided into different categories, and pick the agents which shines in each specific category of the market to construct a pool of policies.

We utilize the pool to train the high-level agent which operates on a minute-level.

We evaluate the high-level agent in the valid and test datasets.


