from src.task import TopicModelTask
from src.components import Conf, GSM
from argparse import ArgumentParser
import torch
from torch.utils.tensorboard import SummaryWriter
from src.dataset import load_news_data, load_kos_data

from os import path

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--evaluate_every", default=5, type=int)
    # parser.add_argument("--max_grad_norm", default=10., type=float)
    parser.add_argument("--batch_size", default=100, type=int)

    parser.add_argument("--data_dir", default="data", type=str)
    parser.add_argument("--runs_dir", default="runs", type=str)
    parser.add_argument("--export_dir", default="export_dir", type=str)

    parser.add_argument("--device", default="cpu", type=str)

    return parser.parse_args()

def write_summary(writer, stat, step, prefix='train'):
    stat = stat.get_dict()
    for k, v in stat.items():
        writer.add_scalars(k, {prefix: v}, global_step=step)


def save_checkpoint(args, model):
    filename = path.join(args.export_dir, 'model_best.pt')
    state_dict = model.to("cpu").state_dict()
    torch.save(state_dict, filename)
    model.to(torch.device(args.device))


def save_topics(args, vocab, topic_prob, epoch, topk=100):
    topic_prob = topic_prob.detach()
    values, indices = torch.topk(topic_prob, k=topk, dim=-1)

    topics = []
    for t in indices:
        topics.append(' '.join([vocab.get_word(i.item()) for i in t]))


    with open(path.join(args.export_dir, "topic-{}.topics".format(epoch)), 'w') as f:
        f.write('\n'.join(topics))

    str_values = []
    for t in values:
        str_values.append(' '.join([str(v) for v in t]))

    with open(path.join(args.export_dir, "topic-{}.values".format(epoch)), 'w') as f:
        f.write('\n'.join(str_values))


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

    conf = Conf(vocab_size=6906, hidden_size=65, latent_size=20)
    model = GSM(conf).to(device)

    task = TopicModelTask("topic model", args)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load datasets
    train_loader, dev_loader, test_loader, vocab = load_kos_data(args, device)

    best_loss = float("inf")
    writer = SummaryWriter(path.join(args.runs_dir, "gsm-3"))

    for e in range(1, args.epochs + 1):
        stats = task.train(model, optim,  train_loader, device)

        print(f"{task.global_step}\t | " + stats.description('train_'))
        write_summary(writer, stats, task.global_step, 'train')

        if e % args.evaluate_every == 0:
            stats = task.eval(model, dev_loader, device)
            print(stats.description('-->eval_'))
            write_summary(writer, stats, task.global_step, 'eval')

            if stats.get_value("loss") < best_loss:
                print("new best model")
                best_loss = stats.get_value("loss")
                save_checkpoint(args, model)
                topics = model.get_topics()
                save_topics(args, vocab, topics, e, topk=10)

