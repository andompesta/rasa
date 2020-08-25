class Conf(object):
    def __init__(
            self,
            vocab_size=6905,
            hidden_size=100,
            latent_size=50,
            topic_size=64,
            dropout_prob=0.2,
            penalty=50,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.penalty = penalty
        self.latent_size = latent_size

        self.topic_size = topic_size

    @property
    def K(self):
        return self.latent_size
