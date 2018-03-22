import json
import shutil
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, f1_score
from os.path import join
from IPython.core.debugger import Pdb

from utils import log


class Tracker:
    def __init__(self, dataset_size, track_loss=True):
        self.track_loss = track_loss
        if track_loss:
            self.loss = 0.0
        self.correct = 0
        self.seen = 0
        self.targets = np.empty(dataset_size, dtype=int)
        self.preds = np.empty(dataset_size, dtype=int)

    def update(self, targets, preds, loss=None):
        if self.track_loss:
            self.loss += loss.data[0]
        self.correct += torch.sum((preds == targets).data)
        self.targets[self.seen:self.seen + targets.size(0)] = targets.data.cpu().numpy()
        self.preds[self.seen:self.seen + targets.size(0)] = preds.data.cpu().numpy()
        self.seen += targets.size(0)

    def print(self, logfile=None):
        acc = float(self.correct) / self.seen * 100
        if self.track_loss:
            loss = self.loss / self.seen
            log('running loss: {:.4f}, running acc: {:2.3f} ({}/{})'.format(loss, acc, self.correct, self.seen), logfile)
        else:
            log('running acc: {:2.3f} ({}/{})'.format(acc, self.correct, self.seen), logfile)
        targets = self.targets[:self.seen]
        preds = self.preds[:self.seen]
        log(confusion_matrix(targets, preds, labels=[0, 1, 2]), logfile)
        log("macro-F1: {:4.4f}".format(f1_score(targets, preds, labels=[0, 1, 2], average='macro')), logfile)

    def getMetrics(self):
        acc = float(self.correct) / self.seen * 100
        targets = self.targets[:self.seen]
        preds = self.preds[:self.seen]
        cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
        fscore = f1_score(targets, preds, labels=[0, 1, 2], average='macro')
        if self.track_loss:
            loss = self.loss / self.seen
            return loss, acc, cm, fscore
        else:
            return acc, cm, fscore


def train(model, dataloader, criterion, optimizer, use_gpu=False, logfile=None):
    model.train()  # Set model to training mode
    tracker = Tracker(len(dataloader.dataset))
    # Iterate over data.
    for step, (reviews, summaries, targets) in enumerate(dataloader):
        if use_gpu:
            targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)

        # zero grad
        optimizer.zero_grad()
        # Pdb().set_trace()
        scores = model(reviews, summaries)
        _, preds = torch.max(scores, 1)
        loss = criterion(scores, targets)

        # backward + optimize
        loss.backward()
        for p in model.parameters():
            if p.grad is None:
                continue
            p.grad.data.clamp_(-0.25, 0.25)
        optimizer.step()

        # statistics
        tracker.update(targets, preds, loss)
        step += 1
        if step % 100 == 0:
            tracker.print(logfile)
        if tracker.seen + dataloader.batch_size > len(dataloader.dataset):
            break
    loss, acc, cm, fscore = tracker.getMetrics()
    log('Train Loss: {:.4f}, Acc: {:2.3f} ({}/{}), macro-F1: {:4.4f}'.format(loss, acc, tracker.correct, tracker.seen, fscore), logfile)
    log(cm, logfile)
    return loss, acc, cm, fscore


def validate(model, dataloader, criterion, use_gpu=False, logfile=None):
    model.eval()  # Set model to evaluate mode
    tracker = Tracker(len(dataloader.dataset))
    for reviews, summaries, targets in dataloader:
        if use_gpu:
            targets = targets.cuda()
        targets = Variable(targets)

        # zero grad
        scores = model(reviews, summaries)
        _, preds = torch.max(scores, 1)
        loss = criterion(scores, targets)

        # statistics
        tracker.update(targets, preds, loss)
    loss, acc, cm, fscore = tracker.getMetrics()
    log('Validation Loss: {:.4f}, Acc: {:2.3f} ({}/{}), macro-F1: {:4.4f}'.format(loss, acc, tracker.correct, tracker.seen, fscore), logfile)
    log(cm, logfile)
    return loss, acc, cm, fscore


def train_model(model, data_loaders, criterion, optimizer, scheduler, save_dir, num_epochs=25, use_gpu=False, best_fscore=0, start_epoch=0, logfile=None):
    log('Training Model with use_gpu={}...'.format(use_gpu), logfile)
    since = time.time()

    best_model_wts = model.state_dict()
    writer = SummaryWriter(save_dir)
    for epoch in range(start_epoch, num_epochs):
        log('Epoch {}/{}'.format(epoch, num_epochs - 1), logfile)
        log('-' * 10, logfile)
        train_begin = time.time()
        train_loss, train_acc, train_cm, train_fscore = train(
            model, data_loaders['train'], criterion, optimizer, use_gpu, logfile)
        train_time = time.time() - train_begin
        log('Epoch Train Time: {:.0f}m {:.0f}s'.format(
            train_time // 60, train_time % 60), logfile)
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Train Accuracy', train_acc, epoch)
        writer.add_scalar('Train Fscore', train_fscore, epoch)

        validation_begin = time.time()
        val_loss, val_acc, val_cm, val_fscore = validate(
            model, data_loaders['val'], criterion, use_gpu, logfile)
        validation_time = time.time() - validation_begin
        log('Epoch Validation Time: {:.0f}m {:.0f}s'.format(
            validation_time // 60, validation_time % 60), logfile)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Validation Accuracy', val_acc, epoch)
        writer.add_scalar('Validation Fscore', val_fscore, epoch)

        # deep copy the model
        is_best = val_fscore > best_fscore
        if is_best:
            best_fscore = val_fscore
            best_model_wts = model.state_dict()

        save_checkpoint(save_dir, {
            'epoch': epoch + 1,
            'acc': val_acc,
            'fscore': val_fscore,
            'state_dict': model.state_dict(),
            # 'optimizer': optimizer.state_dict(),
        }, is_best)

        writer.export_scalars_to_json(save_dir + "/all_scalars.json")
        scheduler.step()

    time_elapsed = time.time() - since
    log('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), logfile)
    log('Best Validation Fscore: {:4f}'.format(best_fscore), logfile)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json(save_dir + "/all_scalars.json")
    writer.close()

    return model


def save_checkpoint(save_dir, state, is_best):
    savepath = join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, join(save_dir, 'model_best.pth.tar'))


def writePreds(outputfile, preds):
    classmap = {0: 1, 1: 3, 2: 5}
    with open(outputfile, 'w') as outf:
        for pred in preds:
            outf.write(str(classmap[pred]) + '\n')


def test_model(model, dataloader, outputfile, use_gpu=False, logfile=None):
    model.eval()  # Set model to evaluate mode
    tracker = Tracker(len(dataloader.dataset), track_loss=False)
    test_begin = time.time()
    outputs = []
    # Iterate over data.
    for reviews, summaries, targets in dataloader:
        if use_gpu:
            targets = targets.cuda()
        targets = Variable(targets)

        scores = model(reviews, summaries)
        _, preds = torch.max(scores, 1)
        outputs.extend([preds.data[i] for i in range(preds.size(0))])
        # statistics
        tracker.update(targets, preds)
    writePreds(outputfile, outputs)
    test_time = time.time() - test_begin
    acc, cm, fscore = tracker.getMetrics()
    log('Test Acc: {:2.3f} ({}/{}), macro-F1: {:4.4f}'.format(acc, tracker.correct, tracker.seen, fscore), logfile)
    log(cm, logfile)
    log('Test Time: {:.0f}m {:.0f}s'.format(test_time // 60, test_time % 60), logfile)
