import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch import nn, Tensor, optim
from typing import Tuple
from src.task.utils import PerplexStatistics



class Task(object):
    def __init__(
            self,
            name: str,
            args,
            global_step: int = 0
    ):
        super(Task, self).__init__()
        self.name = name
        self.args = args
        self.global_step = global_step


    def train(
            self,
            model: nn.Module,
            optimizer: optim.Optimizer,
            dataloader,
            device
    ):
        model.train()
        optimizer.zero_grad()
        steps = 0
        statistics = PerplexStatistics()

        for batch_idx, docs_t in enumerate(dataloader):
            docs_t.to(device)

            with torch.set_grad_enabled(True):
                stats = model(docs_t)
                loss_t = stats["loss"]
                loss_t = loss_t.mean()
                loss_t.backward()

                if "max_grad_norm" in self.args:
                    nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

            # update metrics
            steps += 1
            statistics.add(docs_t, stats)

            if batch_idx % 100:
                torch.cuda.empty_cache()


            # if (steps / self.args.gradient_accumulation_steps) == self.args.steps_per_epoch:
            #     break

        self.global_step += int(steps)
        return statistics

    def eval(
            self,
            model: nn.Module,
            dataloader,
            device,
            **kwargs
    ):
        model.eval()
        statistics = PerplexStatistics()
        steps = 0

        for batch_idx, docs_t in enumerate(dataloader):
            docs_t = docs_t.to(device)

            with torch.set_grad_enabled(False):
                stats = model(docs_t)
            steps += 1
            statistics.add(docs_t, stats)

        return statistics

