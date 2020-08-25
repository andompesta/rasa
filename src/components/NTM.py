import torch
from torch import nn, Tensor
from src.components.utils import kld_normal, topic_covariance_penalty
from src.components import Conf


class NormalParameter(nn.Module):
    def __init__(self, conf):
        super(NormalParameter, self).__init__()
        self.mu = nn.Linear(conf.hidden_size, conf.latent_size)
        self.log_sigma = nn.Linear(conf.hidden_size, conf.latent_size)
        self.reset_parameters()

    def forward(self, hidden_state: Tensor):
        return self.mu(hidden_state), self.log_sigma(hidden_state)

    def reset_parameters(self):
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.zeros_(self.log_sigma.bias)



class EmbTopic(nn.Module):
    def __init__(self, conf):
        super(EmbTopic, self).__init__()
        self.embedding = nn.Embedding(conf.vocab_size, conf.topic_size)
        self.register_parameter("topic_emb", nn.Parameter(torch.Tensor(conf.K, conf.topic_size)))
        self.reset_parameters()

    def forward(self, logit: Tensor):
        # return the log_prob of vocab distribution
        logit = nn.functional.linear(logit, self.topic_emb.T)
        logit = nn.functional.linear(logit, self.embedding.weight)
        return torch.log_softmax(logit, dim=-1)

    def get_topics(self):
        return torch.softmax(nn.functional.linear(self.topic_emb, self.embedding.weight), dim=-1)

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)
        nn.init.normal_(self.topic_emb)

    def extra_repr(self):
        k, d = self.topic_emb.size()
        return 'topic_emb: Parameter({}, {})'.format(k, d)

class GSM(nn.Module):
    def __init__(self, conf: Conf):
        super(GSM, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(conf.vocab_size, conf.hidden_size),
            nn.Sigmoid(),
            nn.Linear(conf.hidden_size, conf.hidden_size),
            nn.Sigmoid(),
            nn.Dropout(conf.dropout_prob, inplace=False)
        )

        self.normal = NormalParameter(conf)

        self.h_to_z = nn.Sequential(
            nn.Linear(conf.latent_size, conf.latent_size),
            nn.Softmax(dim=-1),
            nn.Dropout(conf.dropout_prob, inplace=False)
        )

        self.topics = EmbTopic(conf)

        self.penalty = conf.penalty
        self.reset_parameters()

    def reset_parameters(self):
        for n, p in self.hidden.named_parameters():
            if "linear" in n:
                nn.init.normal_(p, 0.02)

        for n, p in self.hidden.named_parameters():
            if "linear" in n:
                nn.init.normal_(p, 0.02)

    def forward(self, x: Tensor, n_sample=1):
        h = self.hidden(x)

        # compute variational paramters
        mu, log_sigma = self.normal(h)


        rec_loss = 0
        for i in range(n_sample):
            z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
            z = self.h_to_z(z)
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)

        # loss computation
        kld = kld_normal(mu, log_sigma)
        rec_loss = rec_loss / n_sample

        minus_elbo = rec_loss + kld

        penalty, var, mean = topic_covariance_penalty(self.topics.topic_emb)

        return {
            'loss': minus_elbo + penalty * self.penalty,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld,
            'penalty_mean': mean,
            'penalty_var': var,
            'penalty': penalty
        }

    def get_topics(self):
        return self.topics.get_topics()

if __name__ == '__main__':
    gsm = GSM(Conf())
    print(gsm)