import torch
from torch import nn
from typing import Tuple, List


class CNNEncoder(nn.Module):
    def __init__(self, input_height: int, input_width: int, input_channels: int, dropout: float, batch_norm: bool,
                 n_conv: int):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.droput = dropout
        self.batch_norm = batch_norm
        self.n_conv = n_conv

        if self.batch_norm:
            convs = [nn.Conv2d(self.input_channels, 8, 4, padding=1, stride=2), nn.BatchNorm2d(8), nn.ReLU()]
        else:
            convs = [nn.Conv2d(self.input_channels, 8, 4, padding=1, stride=2), nn.ReLU()]
        current_height = self.input_height // 2
        current_width = self.input_width // 2
        current_channels = 8

        for _ in range(1, self.n_conv):
            if self.batch_norm:
                conv = [nn.Conv2d(current_channels, current_channels * 2, 4, padding=1, stride=2),
                        nn.BatchNorm2d(current_channels*2), nn.ReLU()]
            else:
                conv = [nn.Conv2d(current_channels, current_channels * 2, 4, padding=1, stride=2), nn.ReLU()]

            convs.extend(conv)
            current_channels *= 2
            current_height //= 2
            current_width //= 2

        self.last_height = current_height
        self.last_width = current_width

        self.conv = nn.ModuleList(convs)

        self.feature_vec_size = current_channels * current_height * current_width

        if self.batch_norm:
            self.fc = nn.Sequential(nn.Linear(self.feature_vec_size, self.feature_vec_size//4),
                                    nn.BatchNorm1d(self.feature_vec_size//4), nn.ReLU())
        else:
            self.fc = nn.Sequential(nn.Linear(self.feature_vec_size, self.feature_vec_size // 4), nn.ReLU())
        self.feature_vec_size //= 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.conv:
            x = layer(x)

        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        return self.fc(x)


class CNNDecoder(nn.Module):
    def __init__(self, features: int, initial_height: int, initial_width: int, dropout: float, batch_norm: bool,
                 n_conv: int, target_channels: int):
        super().__init__()
        self.features = features
        self.initial_height = initial_height
        self.initial_width = initial_width
        self.droput = dropout
        self.batch_norm = batch_norm
        self.n_conv = n_conv
        self.target_channels = target_channels

        if self.batch_norm:
            self.fc = nn.Sequential(nn.Linear(self.features, self.features*4),
                                    nn.BatchNorm1d(self.features*4), nn.ReLU())
        else:
            self.fc = nn.Sequential(nn.Linear(self.features, self.features*4), nn.ReLU())

        self.view_shape = (self.initial_height, self.initial_width)

        self.channels = 4*self.features//(self.view_shape[0]*self.view_shape[1])

        channels = self.channels

        if self.batch_norm:
            convs = [nn.ConvTranspose2d(channels, channels//2, 4, padding=1, stride=2),
                     nn.BatchNorm2d(channels//2), nn.ReLU()]
        else:
            convs = [nn.ConvTranspose2d(channels, channels//2, 4, padding=1, stride=2), nn.ReLU()]

        current_height = self.initial_height * 2
        current_width = self.initial_width * 2
        current_channels = channels // 2

        for idx in range(1, self.n_conv):
            target_channels = current_channels//2 if idx != self.n_conv-1 else self.target_channels
            if self.batch_norm:
                conv = [nn.ConvTranspose2d(current_channels, target_channels, 4, padding=1, stride=2),
                        nn.BatchNorm2d(target_channels), nn.ReLU()
                        ]
            else:
                conv = [nn.ConvTranspose2d(current_channels, target_channels, 4, padding=1, stride=2), nn.ReLU()]

            convs.extend(conv)
            current_channels //= 2
            current_height *= 2
            current_width *= 2

        self.deconv = nn.ModuleList(convs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(x.shape[0], self.channels, self.view_shape[0], self.view_shape[1])
        for layer in self.deconv:
            x = layer(x)
        return torch.tanh(x)


class ControllerHead(nn.Module):
    def __init__(self, features: int, n_fc: int, n_actions: int, dropout: float, batch_norm: bool):
        super().__init__()
        self.features = features
        self.n_fc = n_fc
        self.n_actions = n_actions
        self.dropout = dropout
        self.batch_norm = batch_norm

        fcs = []
        current_features = self.features
        for _ in range(self.n_fc - 1):
            fc = [nn.Linear(current_features, current_features//2)]
            if self.batch_norm:
                fc.append(nn.BatchNorm1d(current_features // 2))
            fc.append(nn.ReLU())
            if self.dropout > 0:
                fc.append(nn.Dropout(self.dropout))
            fcs.extend(fc)
            current_features //= 2

        fcs.append(nn.Linear(current_features, self.n_actions))

        self.fcs = nn.ModuleList(fcs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.fcs:
            x = layer(x)
        return x


class CNNAgent(nn.Module):
    def __init__(self, input_height: int, input_width: int, input_channels: int, dropout: float, batch_norm: bool,
                 n_conv: int, n_fc: int, n_actions: int):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.droput = dropout
        self.batch_norm = batch_norm
        self.n_conv = n_conv
        self.n_fc = n_fc
        self.n_actions = n_actions

        self.encoder = CNNEncoder(input_height, input_width, input_channels, dropout, batch_norm, n_conv)
        self.controller = ControllerHead(self.encoder.feature_vec_size, self.n_fc-1, self.n_actions, dropout,
                                         batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        output = self.controller(features)
        return output

    def act(self, state: torch.Tensor, epsilon: float, randfloat: torch.Tensor, randint: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if randfloat.item() > epsilon:
                q_value = self.forward(state)
                action = q_value.max(1)[1].item()
            else:
                action = randint
            return action


class VAE(nn.Module):
    # Modified from the example in <https://github.com/pytorch/examples/blob/master/vae/main.py>
    # Not really necessary, but could be used for having an additional loss signal...
    def __init__(self, input_height: int, input_width: int, input_channels: int, dropout: float, batch_norm: bool,
                 n_conv: int, z_size: int):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.n_conv = n_conv
        self.z_size = z_size

        self.encoder = CNNEncoder(self.input_height, self.input_width, self.input_channels, self.dropout,
                                  self.batch_norm, self.n_conv)

        self.fc_mu = nn.Linear(self.encoder.feature_vec_size, self.z_size)
        self.fc_log_var = nn.Linear(self.encoder.feature_vec_size, self.z_size)

        self.fc_z_dc = nn.Linear(self.z_size, self.encoder.feature_vec_size)

        self.decoder = CNNDecoder(self.encoder.feature_vec_size, self.encoder.last_height, self.encoder.last_width,
                                  self.dropout, self.batch_norm, self.n_conv, self.input_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        z = self.fc_z_dc(z)
        return self.decode(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self._reparameterize(mu, log_var)
        return z

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def _reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def only_encode(self):
        del self.fc_z_dc
        del self.decoder


class InverseModel(nn.Module):
    def __init__(self, n_actions: int, state_features: int, hidden_size: int, n_hidden: int, dropout: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        assert n_hidden > 0
        self.n_actions = n_actions
        self.state_features = state_features
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.mlp = nn.Sequential(self._build_layer(self.state_features*2, self.hidden_size),
                                 *[self._build_layer(self.hidden_size, self.hidden_size) for _ in range(self.n_hidden)],
                                 nn.Linear(self.hidden_size, self.state_features))

    def _build_layer(self, features_in: int, features_out: int) -> nn.Module:
        if not self.batch_norm and self.dropout == 0.0:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.ReLU())
        elif self.batch_norm:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.BatchNorm1d(features_out),
                                 nn.ReLU())
        elif dropout > 0.0:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.ReLU(), nn.Dropout(self.dropout))
        else:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.BatchNorm1d(features_out),
                                 nn.ReLU(), nn.Dropout(self.dropout))

    def forward(self, states: torch.Tensor, next_states: torch.Tensor) -> torch.Tensor:
        features = torch.cat((states, next_states), dim=1)
        actions = self.mlp(features)
        return actions


class ForwardModel(nn.Module):
    def __init__(self, n_actions: int, state_features: int, hidden_size: int, n_hidden: int, dropout: float = 0.0,
                 batch_norm: bool = False):
        super().__init__()
        assert n_hidden > 0
        self.n_actions = n_actions
        self.state_features = state_features
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.embedding_size = max(self.n_actions//2, 4)
        self.action_embedding = nn.Embedding(self.n_actions, self.embedding_size)
        self.mlp = nn.Sequential(self._build_layer(self.embedding_size + self.state_features, self.hidden_size),
                                 *[self._build_layer(self.hidden_size, self.hidden_size) for _ in range(self.n_hidden)],
                                 nn.Linear(self.hidden_size, self.state_features))

    def _build_layer(self, features_in: int, features_out: int) -> nn.Module:
        if not self.batch_norm and self.dropout == 0.0:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.ReLU())
        elif self.batch_norm:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.BatchNorm1d(features_out),
                                 nn.ReLU())
        elif dropout > 0.0:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.ReLU(), nn.Dropout(self.dropout))
        else:
            return nn.Sequential(nn.Linear(features_in, features_out), nn.BatchNorm1d(features_out),
                                 nn.ReLU(), nn.Dropout(self.dropout))

    def forward(self, actions: torch.Tensor, states):
        actions = self.action_embedding(actions)
        features = torch.cat((actions, states), dim=1)
        next_states = self.mlp(features)
        return torch.tanh(next_states)


class IntrinsicCuriosity(nn.Module):
    def __init__(self, n_actions: int, icm_state_features: int, icm_hidden_size: int, icm_n_hidden: int,
                 input_height: int, input_width: int, input_channels: int, dropout: float, batch_norm: bool,
                 n_conv: int):
        super().__init__()
        self.n_actions = n_actions
        self.icm_state_features = icm_state_features

        self.icm_hidden_size = icm_hidden_size
        self.icm_n_hidden = icm_n_hidden

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.n_conv = n_conv

        self.encoder = VAE(self.input_height, self.input_width, self.input_channels, self.dropout,
                                  self.batch_norm, self.n_conv, z_size=self.icm_state_features)

        self.encoder.only_encode()

        #  self.encoder = CNNEncoder(self.input_height, self.input_width, self.input_channels, self.dropout,
        #                          self.batch_norm, self.n_conv)

        self.forward_model = ForwardModel(self.n_actions, self.icm_state_features, self.icm_hidden_size,
                                          self.icm_n_hidden)

        self.inverse_model = InverseModel(self.n_actions, self.icm_state_features, self.icm_hidden_size,
                                          self.icm_n_hidden)

        self.states_criterion = nn.MSELoss()
        self.action_criterion = nn.CrossEntropyLoss()

    def forward(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        # Encoding
        encoded_states = self.encoder.encode(states)
        encoded_next_states = self.encoder.encode(next_states)

        # Inverse model
        predicted_actions = self.inverse_model(encoded_states, encoded_next_states)
        actions_loss = self.action_criterion(predicted_actions, actions)

        # Forward model
        predicted_next_states = self.forward_model(actions, encoded_states)

        states_loss = self.states_criterion(predicted_next_states, encoded_next_states)

        return states_loss, actions_loss


if __name__ == '__main__':
    input_height = 256
    input_width = 128
    input_channels = 3
    dropout = 0.5
    batch_norm = True
    n_conv = 5
    z_size = 128
    bs = 32
    x = torch.zeros(bs, input_channels, input_height, input_width)
    vae = VAE(input_height=input_height, input_width=input_width, input_channels=input_channels, dropout=dropout,
              batch_norm=batch_norm, n_conv=n_conv, z_size=z_size)
    y = vae(x)
    print(y.shape)
    n_fc = 3
    n_actions = 8
    agent = CNNAgent(input_height=input_height, input_width=input_width, input_channels=input_channels, dropout=dropout,
                     batch_norm=batch_norm, n_conv=n_conv, n_fc=n_fc, n_actions=n_actions)
    y = agent(x)
    print(y.shape)
