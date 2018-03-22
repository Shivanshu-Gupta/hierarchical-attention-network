import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from os import makedirs
from os.path import join, exists
from IPython.core.debugger import Pdb

# from preprocess import preprocess
from dataset import ReviewsDataset
from train import train_model, test_model
from model import HAN
from utils import log

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml')
parser.add_argument('--testfile', type=str, metavar='PATH')
parser.add_argument('--outputfile', type=str, metavar='PATH')


def load_datasets(config, phases, logfile=None):
    config = config['data']
    log('Loading vocabularies...', logfile)
    import pickle
    review_vocab = pickle.load(open(join(config['dir'], config['review_vocab']), 'rb'))
    if config['review_vocab'] != config['summary_vocab']:
        summary_vocab = pickle.load(open(join(config['dir'], config['summary_vocab']), 'rb'))
    else:
        summary_vocab = review_vocab

    log('Loading preprocessed datasets...', logfile)
    datafiles = {x: config[x]['jsonfile'] for x in phases}
    datasets = {x: ReviewsDataset(data=join(config['dir'], datafiles[x]), review_vocab=review_vocab, summary_vocab=summary_vocab)
                for x in phases}

    def collate_fn(batch):
        reviews = [sample[0] for sample in batch]
        summaries = [sample[1] for sample in batch]
        targets = torch.LongTensor([sample[2] for sample in batch])
        return (reviews, summaries, targets)

    if 'weights' not in config or not config['weights']:
        dataloaders = {x: DataLoader(datasets[x], batch_size=config[x]['batch_size'], shuffle=True if x == 'train' else False, collate_fn=collate_fn) for x in phases}
    else:
        if config['weights'] == 'weighted':
            samplers = {x: datasets[x].get_sampler() if x == 'train' else None for x in phases}
        else:
            samplers = {x: datasets[x].get_sampler(np.array(config['weights'])) if x == 'train' else None for x in phases}
        dataloaders = {x: DataLoader(datasets[x], batch_size=config[x]['batch_size'], shuffle=False, sampler=samplers[x], collate_fn=collate_fn) for x in phases}

    dataset_sizes = {x: len(datasets[x]) for x in phases}
    log(dataset_sizes, logfile)
    log("review vocab size: {}".format(len(review_vocab.itos)), logfile)
    log("summary vocab size: {}".format(len(summary_vocab.itos)), logfile)
    return dataloaders, review_vocab, summary_vocab


def build_model(config, review_vocab, summary_vocab, logfile=None):
    save_dir = config['save_dir']
    use_gpu = config['use_gpu']
    # Create Model
    config['model']['params']['review_vocab_size'] = len(review_vocab)
    config['model']['params']['summary_vocab_size'] = len(summary_vocab)
    config['model']['params']['use_gpu'] = use_gpu
    config = config['model']
    model = HAN(**config['params'])
    log(model, logfile)
    # Copy pretrained word embeddings
    model.review_lookup.weight.data.copy_(review_vocab.vectors)
    if 'combined_lookup' not in config['params']:
        config['params']['combined_lookup'] = False
    if config['params']['use_summary'] and not config['params']['combined_lookup']:
        model.summary_lookup.weight.data.copy_(summary_vocab.vectors)

    # Reload model from checkpoint if provided
    best_fscore = 0
    start_epoch = 0
    if 'reload' in config:
        reloadPath = join(save_dir, config['reload'])
        if exists(reloadPath):
            log("=> loading checkpoint/model found at '{0}'".format(reloadPath), logfile)
            checkpoint = torch.load(reloadPath)
            start_epoch = checkpoint['epoch']
            best_fscore = checkpoint['fscore']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            log("no checkpoint/model found at '{0}'".format(reloadPath), logfile)
    if use_gpu:
        model = model.cuda()
    return model, best_fscore, start_epoch


def main(config):
    logfile = join(config['save_dir'], 'log')
    log(config)
    if config['mode'] == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val']
    dataloaders, review_vocab, summary_vocab = load_datasets(config, phases, logfile)

    # Create Model
    model, best_fscore, start_epoch = build_model(config, review_vocab, summary_vocab, logfile)
    save_dir = config['save_dir']

    if config['mode'] == 'train':
        # Select Optimizer
        if config['optim']['class'] == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                  **config['optim']['params'])
        elif config['optim']['class'] == 'rmsprop':
            optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                      **config['optim']['params'])
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   **config['optim']['params'])
        criterion = nn.CrossEntropyLoss()
        step_size = config['optim']['scheduler']['step']
        gamma = config['optim']['scheduler']['gamma']
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        log("begin training", logfile)
        model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, save_dir,
                            num_epochs=config['training']['n_epochs'], use_gpu=config['use_gpu'],
                            best_fscore=best_fscore, start_epoch=start_epoch, logfile=logfile)
    elif config['mode'] == 'test':
        test_model(model, dataloaders['test'], config['outputfile'], use_gpu=config['use_gpu'], logfile=logfile)
    else:
        log("Invalid config mode %s !!" % config['mode'], logfile)


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    import yaml
    config = yaml.load(open(args.config))
    config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()
    # TODO: seeding still not perfect
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    if args.testfile:
        config['data']['test']['jsonfile'] = args.testfile
        config['outputfile'] = args.outputfile
        config['data']['dir'] = ''
        config['save_dir'] = ''
    else:
        makedirs(config['save_dir'], exist_ok=True)
    main(config)
