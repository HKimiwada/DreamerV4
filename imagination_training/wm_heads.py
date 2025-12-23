# Code to store policy head, value head, reward head for imagination-augmented agents.
# Policy head: outputs action distribution from latent z_t
# Value head: estimates value of z_t (TD target inside imagination)
# Reward head: if the pretrained MineWorld doesn’t predict rewards, train a small reward net