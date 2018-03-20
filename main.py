import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from IPython.core.debugger import Pdb

# from preprocess import preprocess
from dataset import ReviewsDataset
from train import train_model, test_model
from model import HAN

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config.yaml')


def load_datasets(config, phases):
    config = config['data']
    if 'preprocess' in config and config['preprocess']:
        pass
        # print('Preprocessing datasets')
    import pickle
    review_vocab = pickle.load(open(config['dir'] + '/' + config['review_vocab'], 'rb'))
    summary_vocab = pickle.load(open(config['dir'] + '/' + config['summary_vocab'], 'rb'))

    print('Loading preprocessed datasets')
    datafiles = {x: 'audio_{}_dataset.json'.format(x) for x in phases}
    datasets = {x: ReviewsDataset(data=config['dir'] + '/' + datafiles[x], review_vocab=review_vocab, summary_vocab=summary_vocab)
                for x in phases}
    samplers = {x: datasets[x].get_sampler() for x in phases}

    def collate_fn(batch):
        documents = [sample[0] for sample in batch]
        targets = torch.LongTensor([sample[1] for sample in batch])
        return (documents, targets)
    dataloaders = {x: DataLoader(datasets[x], batch_size=config[x]['batch_size'], shuffle=False, sampler=samplers[x], collate_fn=collate_fn) for x in phases}

    dataset_sizes = {x: len(datasets[x]) for x in phases}
    print(dataset_sizes)
    print("review vocab size: {}".format(len(review_vocab.itos)))
    print("summary vocab size: {}".format(len(summary_vocab.itos)))
    return dataloaders, review_vocab, summary_vocab


def main(config):
    if config['mode'] == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val']
    dataloaders, review_vocab, summary_vocab = load_datasets(config, phases)

    # add model parameters to config
    config['model']['params']['vocab_size'] = len(review_vocab)
    config['model']['params']['use_gpu'] = config['use_gpu']
    model = HAN(**config['model']['params'])
    print(model)
    criterion = nn.CrossEntropyLoss()

    if config['optim']['class'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              **config['optim']['params'])
    elif config['optim']['class'] == 'rmsprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                  **config['optim']['params'])
    else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               **config['optim']['params'])

    best_acc = 0
    # Pdb().set_trace()
    startEpoch = 0
    if 'reload' in config['model']:
        pathForTrainedModel = os.path.join(config['save_dir'],
                                           config['model']['reload'])
        if os.path.exists(pathForTrainedModel):
            print(
                "=> loading checkpoint/model found at '{0}'".format(pathForTrainedModel))
            checkpoint = torch.load(pathForTrainedModel)
            startEpoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
    if config['use_gpu']:
        model = model.cuda()

    print('config mode ', config['mode'])
    save_dir = os.path.join(os.getcwd(), config['save_dir'])

    if config['mode'] == 'train':
        print('lr_scheduler.StepLR')
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        print("begin training")
        model = train_model(model, dataloaders, criterion, optimizer, exp_lr_scheduler, save_dir,
                            num_epochs=config['training']['n_epochs'], use_gpu=config['use_gpu'], best_accuracy=best_acc, start_epoch=startEpoch)
    elif config['mode'] == 'test':
        outputfile = os.path.join(save_dir, config['mode'] + ".json")
        test_model(model, dataloaders['test'], outputfile, use_gpu=config['use_gpu'])
    else:
        print("Invalid config mode %s !!" % config['mode'])


if __name__ == '__main__':
    global args
    args = parser.parse_args()
    import yaml
    config = yaml.load(open(args.config))
    config['use_gpu'] = config['use_gpu'] and torch.cuda.is_available()

    # TODO: seeding still not perfect
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    main(config)
