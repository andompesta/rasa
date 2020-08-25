import torch


class RunningStatistics:

    def __init__(self):
        self.statistics = None

    def add(self, stat):

        if self.statistics is None:
            self.statistics = {}
            for k, v in stat.items():
                if isinstance(v, dict):
                    assert 'value' in v and 'weight' in v, 'value weight should be in {}'.format(str(v))
                    self.statistics[k] = v
                else:
                    self.statistics[k] = {'value': v, 'weight': 1}
        else:
            for k, v in stat.items():
                if not isinstance(v, dict):
                    v = {'value': v, 'weight': 1}
                last_v = self.statistics[k]['value']
                last_w = self.statistics[k]['weight']
                self.statistics[k]['value'] = (v['value'] * v['weight'] + last_v * last_w) / (last_w + v['weight'])
                self.statistics[k]['weight'] += v['weight']
        return self

    def description(self, prefix=''):
        if self.statistics is None:
            return 'None'
        return ' | '.join(['{} v{:.6f} w{:.6f}'.format(prefix + k, v['value'], v['weight'])
                           for k, v in self.statistics.items()])

    def get_value(self, k):
        return self.statistics[k]['value']

    def get_dict(self):
        if self.statistics is None:
            return {}
        return {k: v['value'] for k, v in self.statistics.items()}



class WeightedSum:
    def __init__(self, name, init, postprocessing=None):
        self.name = name
        self.v = init
        self.w = 0
        self.postprocessing = postprocessing

    def add(self, v, w):
        self.v = self.v + v * w
        self.w = self.w + w

    def get(self):
        # No accumulated value
        if self.w == 0:
            return 0
        v = self.v / self.w
        if self.postprocessing is not None:
            v = self.postprocessing(v)
        return v

    def __repr__(self):
        return '{} {:.5f}'.format(self.name, self.get())


class PerplexStatistics:

    def __init__(self):

        def _item(x):
            return x.item()

        def _exp_item(x):
            return torch.exp(x).item()

        self.stat = {
            'ppx': (WeightedSum('ppx', 0, _exp_item), '', ''),
            'ppx_doc': (WeightedSum('ppx_doc', 0, _exp_item), '', ''),
            'loss': (WeightedSum('loss', 0, _item), 'loss', 'doc_count'),
            'loss_rec': (WeightedSum('loss_rec', 0, _item), 'rec_loss', 'doc_count'),
            'kld': (WeightedSum('kld', 0, _item), 'kld', 'doc_count'),
            'penalty': (WeightedSum('penalty', 0, _item), 'penalty', 'doc_count'),
            'penalty_mean': (WeightedSum('penalty_mean', 0, _item), 'penalty_mean', 'doc_count'),
            'penalty_var': (WeightedSum('penalty_var', 0, _item), 'penalty_var', 'doc_count'),
        }

    def add(self, data_batch, stat):
        """Accumulate statistics."""
        with torch.no_grad():
            weight = {
                'word_count': data_batch.sum(),
                'doc_count': len(data_batch)
            }

            for s, k, w in self.stat.values():
                if s.name == 'ppx_doc':
                    s.add((stat['minus_elbo'] / data_batch.sum(dim=-1)).sum() / weight['doc_count'],
                          weight['doc_count'])
                elif s.name == 'ppx':
                    s.add(stat['minus_elbo'].sum() / weight['word_count'], weight['word_count'])
                else:
                    if k not in stat:  # skip for compatibility of multiple models.
                        continue
                    s.add(stat[k].mean(), weight[w])
        return self

    def description(self, prefix=''):
        return ' | '.join(['{} {:.5f}'.format(prefix + k, v)
                           for k, v in self.get_dict().items()])

    def get_value(self, k):
        """Get the accumulated value."""
        return self.stat[k][0].get()

    def get_dict(self):
        r = {}
        for k in self.stat.keys():
            t = self.stat[k][0].get()
            if t != 0:
                r[k] = t
        return r