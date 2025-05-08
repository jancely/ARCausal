import torch
import torch.nn as nn


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


class Model(nn.Module):
    def __init__(self, configs, base_model="GRU"):
        # def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU"):
        super().__init__()

        self.d_feat = configs.enc_in
        self.hidden_size = configs.d_model
        self.use_norm = configs.use_norm
        self.pred_len = configs.pred_len

        if base_model == 'GRU':
            self.encoder = nn.GRU(
                input_size=configs.enc_in,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
                dropout=configs.dropout,
            )
        else:
            self.encoder = nn.LSTM(
                input_size=configs.enc_in,
                hidden_size=configs.d_model,
                num_layers=configs.e_layers,
                batch_first=True,
                dropout=configs.dropout,
            )

        self.fc_z_mu = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.LeakyReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )

        self.fc_z_logvar = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.LeakyReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )

        self.out = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model),
            nn.LeakyReLU(),
            nn.Linear(configs.d_model, configs.enc_in)
        )

        self.fc_out = nn.Linear(configs.seq_len, configs.pred_len)

    def forward_y(self, x, z):
        z = z.repeat(1, x.shape[1], 1)
        z = torch.cat([x, z], dim=-1)

        pred_all = self.out(z)
        # pred_all = self.out(z).squeeze().reshape([x.shape[0], -1])
        return pred_all

    def forward(self, x_mark, batch_x_mark, dec_inp, batch_y_mark):
        if self.use_norm:
            means = x_mark.mean(1, keepdim=True).detach()
            x_mark = x_mark - means
            stdev = torch.sqrt(torch.var(x_mark, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_mark /= stdev

        x = x_mark
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        x = x.reshape(len(x), self.d_feat, -1)
        x = x.permute(0, 2, 1)
        features, _ = self.encoder(x)
        features = features[:, -1, :]

        features = features.reshape([x_mark.shape[0], x_mark.shape[1], -1])
        # print('features', features.shape)
        context_feas = features.mean(1, keepdim=True)
        mu = self.fc_z_mu(context_feas)
        logvar = self.fc_z_logvar(context_feas)

        pred_all = []
        for i in range(5):
            z = reparameterize(mu, logvar)
            pred = self.forward_y(features, z)
            pred_all.append(pred)
        output = torch.stack(pred_all).mean(0)
        # output = torch.stack(pred_all).mean(0)

        output = self.fc_out(output.transpose(1, 2)).transpose(1, 2)

        if self.use_norm:
          output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
          output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # print(output.shape)

        return output
