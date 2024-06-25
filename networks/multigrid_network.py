import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_DIRECTIONS = 4

class MultiGridNetwork(nn.Module):
    def __init__(self, obs, config, n_actions, n_agents, agent_id):
        """
        Initialize Deep Q Network

        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(MultiGridQNetworkKamal, self).__init__()
        self.obs_shape = obs
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.config = config
        self.agent_id = agent_id

        self.image_layers = nn.Sequential(
            nn.Conv2d(3, 32, (self.config.kernel_size, self.config.kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (self.config.kernel_size, self.config.kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),  # [B, 64, 1, 1] -> [B, 64]
            nn.Linear(64, 64),  
            nn.LeakyReLU()
            )

        self.direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS * self.n_agents, self.config.fc_direction),
            nn.ReLU(),
            )

        #interm = (obs['image'].shape[1]-self.config.kernel_size)+1
        self.head = nn.Sequential(
            nn.Linear(64 + self.config.fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_actions),
        )

    def process_image(self, x):
        if len(x.shape) == 3:
            # Add batch dimension
            x = x.unsqueeze(0)
            
        # Change from (B,H,W,C) to (B,C,W,H) (i.e. RGB channel of dim 3 comes first)
        x = x.permute((0, 3, 1, 2))
        x = x.float()
        return x
            
    def forward(self, obs):
        # process image
        x = torch.tensor(obs['image']).to(device)
        x = self.process_image(x)
        batch_dim = x.shape[0]

        # Run conv layers on image
        image_features = self.image_layers(x)
        image_features = image_features.reshape(batch_dim, -1)

        # Process direction and run direction layers
        dirs = torch.tensor(obs['direction']).to(device)
        if batch_dim == 1:  # 
            dirs = torch.tensor(dirs).unsqueeze(0)
        dirs_onehot = torch.nn.functional.one_hot(dirs.to(torch.int64), num_classes=NUM_DIRECTIONS).reshape((batch_dim, -1)).float()
        dirs_encoding = self.direction_layers(dirs_onehot)

        # Concat
        features = torch.cat([image_features, dirs_encoding], dim=-1)

        # Run head
        return self.head(features)