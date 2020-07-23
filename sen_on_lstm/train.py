import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from data_loader import *
import model
from utils import repackage_hidden
import os
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--chunk_size', type=int, default=10,
                    help='number of units per chunk')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')

parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')

parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
# parser.add_argument('--batch_size', type=int, default=60, metavar='N',
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.45,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')

#####################################################################################
parser.add_argument('--dropoute', type=float, default=0.,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.,
                    help='dropout for input embedding layers (0 = no dropout)')
#####################################################################################


parser.add_argument('--wdrop', type=float, default=0.45,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=141,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default='../ckpt_19_l2.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument('--finetuning', type=int, default=500,
                    help='When (which epochs) to switch to finetuning')
parser.add_argument('--philly', action='store_true',
                    help='Use philly cluster')
args = parser.parse_args()
args.tied = True

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    if args.philly:
        fn = os.path.join(os.environ['PT_OUTPUT_DIR'], fn)
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


###############################################################################
# Training code
###############################################################################

def train(model, train_dataloader, epoch):
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    model.train()
    total_loss = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    hidd = model.init_hidden(args.batch_size*19)
    hidd_cand = model.init_hidden(args.batch_size*4)
    batch = 0
    acc_list = []
    total_var = 0

    it = tqdm(range(len(train_dataloader)), desc="Epoch {}/{}".format(epoch, args.epochs), ncols=0)
    data_iter = iter(train_dataloader)
    for niter in it:
        input_ids, cand_ids, target = data_iter.next()
        if args.cuda:
            input_ids = input_ids.cuda()
            cand_ids = cand_ids.cuda()
            targets = target.cuda()

        hidden = repackage_hidden(hidden)
        hidd = repackage_hidden(hidd)
        hidd_cand = repackage_hidden(hidd_cand)
        optimizer.zero_grad()

        output, result_prob, hidden, rnn_hs, dropped_rnn_hs, cand_emb = model(input_ids, cand_ids, hidden, hidd, hidd_cand)
        distance_1 = model.distance[0][2]
        dis_var = torch.var(distance_1)

        l2_loss = criterion_var(dis_var, torch.tensor(0.2).cuda())
        raw_loss = criterion(result_prob, targets)

        _, predict = result_prob.max(dim=-1)
        acc = float(torch.sum(predict == targets)) / float(targets.size(0))
        acc_list.append(acc)

        # loss = raw_loss
        loss = raw_loss + l2_loss
        # Activiation Regularization
        if args.alpha:
            loss = loss + sum(
                args.alpha * dropped_rnn_h.pow(2).mean()
                for dropped_rnn_h in dropped_rnn_hs[-1:]
            )
        # Temporal Activation Regularization (slowness)
        if args.beta:
            loss = loss + sum(
                args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in rnn_hs[-1:]
            )
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        total_var += dis_var.data


    cur_loss = total_loss / len(train_dataloader)
    cur_var = total_var / len(train_dataloader)
    elapsed = time.time() - start_time
    print('| epoch {:3d} | lr {:05.5f} | ms/batch {:5.2f} | '
          'loss {:5.2f} | acc {:8.4f} | var {:8.4f}'.format(
        epoch, optimizer.param_groups[0]['lr'],
        elapsed * 1000 / len(train_dataloader), cur_loss, np.mean(acc_list), cur_var))


def evaluate(val_dataloader, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    hidd = model.init_hidden(args.batch_size*19)
    hidd_cand = model.init_hidden(args.batch_size*4)
    acc_list = []

    it = tqdm(range(len(val_dataloader)), desc="Eval", ncols=0)
    data_iter = iter(val_dataloader)
    cur_var = 0

    for niter in it:

        input_ids, cand_ids, target = data_iter.next()
        if args.cuda:
            input_ids = input_ids.cuda()
            cand_ids = cand_ids.cuda()
            targets = target.cuda()

        hidden = repackage_hidden(hidden)
        hidd = repackage_hidden(hidd)
        hidd_cand = repackage_hidden(hidd_cand)
        optimizer.zero_grad()

        output, result_prob, hidden, rnn_hs, dropped_rnn_hs, cand_emb = model(input_ids, cand_ids, hidden, hidd, hidd_cand)
        distance_1 = model.distance[0][2]
        dis_var = torch.var(distance_1)
        l2_loss = criterion_var(dis_var, torch.tensor(0.2).cuda())
        raw_loss = criterion(result_prob, targets)

        _, predict = result_prob.max(dim=-1)
        acc = float(torch.sum(predict == targets)) / float(targets.size(0))
        acc_list.append(acc)
        total_loss += raw_loss.data + l2_loss.data
        cur_var += dis_var.data
        hidden = repackage_hidden(hidden)
    return total_loss / len(val_dataloader), np.mean(acc_list), cur_var/len(val_dataloader)


if __name__ == '__main__':

    train_dataloader = get_loader(phase='train', batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_dataloader = get_loader(phase='val', batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
   
    ###############################################################################
    # Build the model
    ###############################################################################

    dict_file_name = os.path.join('../../../Ordered-Neurons/recipe_data/', 'dict_recipe.pkl')
    dictionary = pickle.load(open(dict_file_name, 'rb'))
    model = model.RNNModel(args.model, len(dictionary), args.emsize, args.nhid, args.chunk_size, args.nlayers,
                           args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

    criterion = nn.CrossEntropyLoss()
    criterion_var = nn.MSELoss()

    ###
    if args.resume:
        print('Resuming model ...')
        model_load(args.resume)
        optimizer.param_groups[0]['lr'] = args.lr
        model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
        if args.wdrop:
            for rnn in model.rnn.cells:
                rnn.hh.dropout = args.wdrop
    ###
    if args.cuda:
        model = model.cuda()
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0, 0.999), eps=1e-9, weight_decay=args.wdecay)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, patience=2, threshold=0)
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            val_loss2, acc, dis_var = evaluate(val_dataloader, args.batch_size)
            print(acc)
            train(model, train_dataloader, epoch)

            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    # prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2, acc, dis_var = evaluate(val_dataloader, args.batch_size)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid acc {:8.4f} | var {:8.4f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, acc, dis_var))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(args.save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

                if epoch == args.finetuning:
                    print('Switching to finetuning')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    best_val_loss = []

                if epoch > args.finetuning and len(best_val_loss) > args.nonmono and val_loss2 > min(
                        best_val_loss[:-args.nonmono]):
                    print('Done!')
                    import sys

                    sys.exit(1)

            else:
                val_loss, acc, dis_var = evaluate(val_dataloader, args.batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid acc {:8.4f} | var {:8.4f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, acc, dis_var))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(args.save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if args.optimizer == 'adam':
                    scheduler.step(val_loss)

                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(args.save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

            print("PROGRESS: {}%".format((epoch / args.epochs) * 100))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

