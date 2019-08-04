from __future__ import print_function

import os
import time
import argparse
import random
import torch
import torch.optim as optim
from torch.autograd import Variable

from data_loader import loadTrainData, loadTestData
from evaluation import evaluate_pred
from model import dynemb
'''
# Utils to import as required
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
'''

parser = argparse.ArgumentParser(description='Code')
parser.add_argument('--dataset', type=str, default='M1', help='SO | Generic ')
parser.add_argument('--dataroot', type=str, default='./data/Social Evolution/', help='location of the data')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nodes', type=int, default=100, help='number of nodes in graph')
parser.add_argument('--ndyn', type=int, default=2, help='number of different scale dynamics in graph')
parser.add_argument('--eval_batch_size', type=int, default=1, metavar='N', help='test batch size')
parser.add_argument('--batch', type=int, default=300, help='train batch size')
parser.add_argument('--emsize', type=int, default=32, help='size of node embeddings')
parser.add_argument('--nhid', type=int, default=32, help='number of hidden units per layer')
parser.add_argument('--nsamples', type=int, default=5, help='number of samples for survival computation')
parser.add_argument('--samples', type=int, default=10, help='number of samples for prediction')
parser.add_argument('--testsamples', type=int, default=1000, help='number of test samples for smaller test')

parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate, default=0.0002')
parser.add_argument('--clip', type=float, default=10, help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')

parser.add_argument('--phase', type=str,  default='1', help='Train_and_Test = 0, Train=1')
parser.add_argument('--save', type=str,  default='./output_main/', help='folder to save model checkpoints and final model')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')

opt = parser.parse_args()

try:
    os.makedirs(opt.save)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)


def train(model, optimizer, train_data, batch_num, N, lr):
    model.train()
    total_loss = 0.0
    epoch_loss = 0.0
    start_time = time.time()

    for i in range(batch_num):
        mini_batch = Variable(train_data[i])
        mini_batch_node_list = []
        for j in range(mini_batch.size(0)):
            inp_tuple = mini_batch[j]
            v1 = int(inp_tuple.data[0])
            v2 = int(inp_tuple.data[1])
            mini_batch_node_list.append(v1)
            mini_batch_node_list.append(v2)

        mini_batch_node_list = list(set(mini_batch_node_list))

        optimizer.zero_grad()
        output_intensity, output_survival = model(mini_batch, mini_batch_node_list, N)
        intensity_loss = torch.sum(torch.log(outputs_intensity))
        survival_loss = torch.sum(outputs_survival)
        mini_batch_loss = -intensity_loss + survival_loss

        mini_batch_loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()
        total_loss += mini_batch_loss.data

        if i % opt.log_interval == 0 and i > 0:
            cur_loss = total_loss[0] / opt.log_interval
            epoch_loss += cur_loss
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:3d}/{:3d} batches | lr {:2.4f} | Training Time {:5.2f} s/batch | '
                    'Current Training Loss {:5.2f}'.format(
                epoch + 1, i, batch_num, lr,
                elapsed / opt.log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
    epoch_loss = epoch_loss / (batch_num / opt.log_interval)
    return epoch_loss


if __name__== '__main__':

    if opt.phase == '0':
        print('Phase 0 - Both Training and Testing.')
    if opt.phase == '1':
        print('Phase 1 - Only Training. No Testing.')

    N = opt.nsamples
    N_t = opt.samples
    if opt.phase in ['0', '1']:

        print('\nLoading Data...')
        train_data, train_batch_num, train_inter_dict, train_link_dict = loadTrainData(opt.dataroot, opt.nodes, opt.batch)

        print('Initializing model and optimizer...\n')
        dynemb = dynemb(opt.nodes, opt.emsize, opt.ndyn, opt.eval_batch_size)

        lr = opt.lr
        optimizer = optim.Adam(dynemb.parameters(), lr=lr)
        # Training Module

        print('Total number of train batches: ', train_batch_num)
        try:
            for epoch in range(0, opt.epochs):
                epoch_start_time = time.time()
                epoch_loss = train(dynemb, optimizer, train_data, train_batch_num, A, S, N, lr)
                print('-' * 89)
                print('| End of Epoch {:3d} | Time Elapsed: {:5.2f}s | Epoch Loss {:5.6f} | ' .format(epoch,
                                                                        (time.time() - epoch_start_time), epoch_loss))
                print('-' * 89)
        except KeyboardInterrupt:
            print('Exiting Training Early due to Keyboard Interrupt')

        model_file = opt.save + 'M1_model_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + '.pt'
        train_l = opt.save + 'M1_model_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + 'tdl.pkl'
        train_i = opt.save + 'M1_model_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + 'tdi.pkl'
        torch.save(dynemb, model_file)
        torch.save(train_link_dict, train_l)
        torch.save(train_inter_dict, train_i)

    if opt.phase == '0':

        start_time = time.time()
        model_file = opt.save + 'M1_model_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + '.pt'
        l_file = opt.save + 'M1_model_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + 'tdl.pkl'
        i_file = opt.save + 'M1_model_epochs_' + str(opt.epochs) + '_lr_' + str(opt.lr) + 'tdi.pkl'
        dynemb = torch.load(model_file)
        l_dict = torch.load(l_file)
        i_dict = torch.load(i_file)

        test_data, test_batch_num = loadTestData(opt.dataroot, opt.eval_batch_size)
        mae_inter, rmse_inter = evaluate_pred(dynemb, test_data, test_batch_num,
                                                                                 N, N_t, opt.testsamples)
        print('-' * 80)
        print("Total Test Time: {:5.2f}s \n".format(time.time() - start_time))

        for i in range(6):
            print('| Testing Batch: {:5.2f}s \n| '
                  'MAE: {:5.6f} \n| RMSE: {:5.6f} '
                  .format(i, mae_inter[i], rmse_inter[i]))
        print('-' * 80)
