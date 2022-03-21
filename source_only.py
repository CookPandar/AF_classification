from __future__ import division
import os
import json
import pickle
import random
import wfdb
import math
import numpy as np

np.set_printoptions(suppress=True)
import warnings
import traceback
import time
from time import strftime, localtime
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, \
    roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score
from scipy import stats
from decimal import Decimal
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import MIAT_utils as utils
from torch.backends import cudnn
import torch.nn as nn
from torch.autograd import Variable
from tsne import vis
import block_network

cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser()
parser.add_argument("--lr", dest="init_lr", type=float, metavar='<float>', default=0.0005)
parser.add_argument("--class", dest="class_num", type=int, metavar='<int>', default=5)
parser.add_argument("--alpha", dest="mix_alpha", type=float, metavar='<float>', default=1.)
parser.add_argument("--drop", dest="drop_keep", nargs='+', type=float, metavar='<float>',default=[0.5,0.5,0.3])
parser.add_argument("--lambda", dest="_lambda", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lambda_i", dest="inconsist", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lambda_e", dest="ent", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lambda_t", dest="t_vat", type=float, metavar='<float>', default=0.001)
parser.add_argument("--focal", dest="use_focal", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--decay", dest="use_decay", type=str2bool, metavar='<bool>', default=True)
parser.add_argument("--logit", dest="logit", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--weight", dest="weights_decay", type=float, metavar='<float>', default=0.0005)
parser.add_argument("--run_id", dest="id", type=int, metavar='<int>', default=0)
parser.add_argument("--n", dest="nb_epoch", type=int, metavar='<int>', default=0)
parser.add_argument("--N", dest="G_update", type=int, metavar='<int>', default=1)
parser.add_argument("--epochs", dest="EPOCHS", type=int, metavar='<int>', default=150)
parser.add_argument("--num", dest="init_cand_num", type=int, metavar='<int>', default=1000)
parser.add_argument('--gpu', dest="GPU", type=str, default=7, help='cuda_visible_devices')
parser.add_argument("--mcd", dest="MCD", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--mix", dest="mixup", type=str2bool, metavar='<bool>', default=True)
parser.add_argument("--vat", dest="VAT", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--tsne", dest="TSNE", type=str2bool, metavar='<bool>', default=False)
parser.add_argument('--s', dest="source", type=str, default='DS1')
parser.add_argument('--t', dest="target", type=str, default='DS2')
args = parser.parse_args()


cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)
random.seed(666)
np.random.seed(666)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 128

N_CLASS = args.class_num
N = args.G_update
alpha = 0.6
num_steps = 500
nb_epoch = args.nb_epoch
beta = (0.9, 0.999)


tmp_dir = 'exp0/'
weight_path = 'exp0/weights/'
train_results_save_path = 'exp0/train_results/'
train_results_img_save_path = 'exp0/train_results/imgs/'

if not (os.path.exists(train_results_img_save_path)):
    os.makedirs(train_results_img_save_path)
if not (os.path.exists(train_results_save_path)):
    os.makedirs(train_results_save_path)
if not (os.path.exists(weight_path)):
    os.makedirs(weight_path)

with open('exp0/' + 'run_' + str(args.id) + '_results.txt', 'a') as f:
    f.write('\n\n==============' + __file__ + '====================\n')
    f.write(args.__str__() + '\n')

print('N_class:', N_CLASS)

x_train, y_train, x_val, y_val, x_test, y_test, x_target, y_target, class_center = utils.get_dataset(args.source,
                                                                                        args.target, n_class=N_CLASS)
label_target = torch.zeros((x_target.shape[0])).long()


def train(train_x, train_y, target_x, model, optimizer, epoch, w_loss_ratio=None, new_data=None,
            new_label=None, new_label_index=None, target_weights=None,val_f1_class=None):

    G.train()
    Ft.train()

    p1_loss = []
    p2_loss = []
    pt_loss = []
    w_diff_loss = []
    total_loss = []
    p1_acc = []
    p2_acc = []
    pt_acc = []

    gen_source_only_batch = utils.batch_generator([train_x, train_y], batch_size//2)

    # gen_all_target_batch = utils.batch_generator([x_target, label_target], batch_size//2, shuffle=True)

    num_steps = train_x.shape[0] // batch_size * 2 + 1

    for i in range(num_steps):

        x0, y0 = gen_source_only_batch.__next__()
        x0, y0 = x0.to(device), y0.to(device)

        zero_grad()
        # w_loss = torch.tensor([0.]).float().to(device)
        if args.mixup:
            lam = np.random.beta(args.mix_alpha, args.mix_alpha)
            lam = max(lam, 1 - lam)
            index = np.random.permutation(x0.shape[0])
            x_0, y_0 = x0[index], y0[index]
            mix_feature = lam * G(x0) + (1 - lam) * G(x_0)
            mix_input = lam * x0 + (1 - lam) * x_0
            mix_output = G(mix_input)
            w_loss = args._lambda * F.mse_loss(mix_feature, mix_output)

            if args.VAT:
                mixed_x = lam * x0 + (1 - lam) * x_0
                feature_t = G(mixed_x)
                pred_t = Ft(feature_t)
                pred_t_loss = lam * loss_func(pred_t, y0) + (1 - lam) * loss_func(pred_t, y_0) + w_loss

        else:
            w_loss = torch.tensor([0.]).float().to(device)
            feature = G(x0)
            pred_t = Ft(feature)
            pred_t_loss = loss_func(pred_t, y0) + w_loss

        pred_t_loss.backward()
        optimizer_F.step()
        optimizer_Ft.step()

        pred_label_t = torch.argmax(pred_t, dim=1)
        pred_t_acc = (pred_label_t == y0).float().mean().detach().cpu().numpy()

        pt_loss.append(pred_t_loss.item())
        pt_acc.append(pred_t_acc)


    print('---Num_steps: {}, S_samples: {}, T_nums: {}, S+T_samples: {}, '
          ' Source_batch: {} '.format(num_steps, train_x.shape[0], 0,
                                      train_x.shape[0], x0.shape[0]))

    return np.mean(pt_loss), np.mean(pt_acc)


def val(val_x, val_y, model, epoch, best_val_f1, w_loss_ratio=None, alpha=None):

    G.eval()
    Ft.eval()

    with torch.no_grad():

        p1_loss = []
        p2_loss = []
        pt_loss = []
        w_diff_loss = []
        total_loss = []
        p1_acc = []
        p2_acc = []
        pt_acc = []
        size_val = 0
        y_true = np.array([]).reshape((0, 1))
        y_pred = np.array([]).reshape((0, 1))

        gen_val_batch = utils.batch_generator(
            [val_x, val_y], batch_size*4, test=True)
        num_iter = int(val_x.shape[0] // (batch_size*4)) + 1
        step = 0

        while step < num_iter:
            x1, y1 = gen_val_batch.__next__()
            x1, y1 = x1.to(device), y1.to(device)

            features = G(x1)
            pred_t = Ft(features)
            pred_t_loss = loss_func(pred_t, y1)
            pred_label_t = torch.argmax(pred_t, dim=1)
            pred_t_acc = (pred_label_t == y1).float().mean().detach().cpu().numpy()

            pred = pred_label_t.detach().cpu().numpy().reshape(y1.shape[0],1)
            label = y1.detach().cpu().numpy().reshape(y1.shape[0], 1)
            y_pred = np.concatenate((y_pred, pred), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)

            pt_loss.append(pred_t_loss.item())
            pt_acc.append(pred_t_acc)
            size_val += x1.shape[0]

            step += 1

        print('---num_iter: {}, size_val: {}'.format(num_iter, size_val))

        Val_f1 = f1_score(y_true, y_pred, average=None)

        val_f1 = np.mean(Val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'G_state_dict': G.state_dict(),
                'Ft_state_dict': Ft.state_dict(),
                'epoch': epoch,
                'best val f1': best_val_f1,
                 }, weight_path + str(id) + '_torch_best_f1_source_only_model.pt')

            print("best f1: {:.4f}".format(best_val_f1))
        else:
            print("=======val f1: {:.4f}, but not the best f1======== ".format(val_f1))

    return np.mean(pt_loss), np.mean(pt_acc), best_val_f1, val_f1, Val_f1


def test(test_x, test_y, model, domain='target_test'):

    print('Load model...')
    checkpoint = torch.load(weight_path + str(id) + '_torch_best_f1_source_only_model.pt')
    print('epoch:', checkpoint['epoch'])
    print('best val f1:', checkpoint['best val f1'])

    G.load_state_dict(checkpoint['G_state_dict'])
    Ft.load_state_dict(checkpoint['Ft_state_dict'])
    G.eval()
    Ft.eval()

    with torch.no_grad():
        y_true = np.array([]).reshape((0, 1))
        y_pred = np.array([]).reshape((0, 1))
        feature = np.array([]).reshape((0, 2560))
        gen_target_batch = utils.batch_generator(
            [test_x, test_y], batch_size*4, test=True)
        num_iter = int(test_x.shape[0] // (batch_size*4)) + 1
        print('---test_num_iter:', num_iter)
        step = 0
        while step < num_iter:
            x1, y1 = gen_target_batch.__next__()
            x1, y1 = x1.to(device), y1.to(device)
            fea = G(x1)
            pred_t = Ft(fea)
            pred_label_t = torch.argmax(pred_t, dim=1)

            pred = pred_label_t.detach().cpu().numpy().reshape(y1.shape[0], 1)
            label = y1.detach().cpu().numpy().reshape(y1.shape[0], 1)
            fea = fea.detach().cpu().numpy().reshape(y1.shape[0], -1)
            y_pred = np.concatenate((y_pred, pred), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)
            feature = np.concatenate((feature, fea), axis=0)
            step += 1

    if domain != 'target_train':
        print(classification_report(y_true, y_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))


    print('========== confusion matrix ==========')
    print('domian:', domain)
    print(confusion_matrix(y_true, y_pred))

    # print('===Save features for tSNE !!=======')
    # np.savez(train_results_save_path + 'exp_' + str(args.id) + "_" + domain + '_fea' , feature=feature, label=y_true)

    if domain != 'target_train':
        with open('exp0/' + 'run_' + str(args.id) + '_results.txt', 'a') as f:
            f.write('best epoch:'+ str(checkpoint['epoch']) + '  ' + 'best val f1:' + str(checkpoint['best val f1']) + '\n')
            f.write(classification_report(y_true, y_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4) + '\n')
            f.write(str(confusion_matrix(y_true, y_pred)) + '\n')


def plot_statistic(train_statistic, val_statistic, i=0):

    iters = [i + 1 for i in range(0, len(train_statistic['pt_loss']))]

    # plt.figure(figsize=(10, 10))
    # plt.plot(iters, train_statistic['w_diff_loss'], color='red', label='weight_diff_loss')
    # plt.plot(iters, train_statistic['p1_acc'], color='green', label='train_p1_acc')
    # plt.plot(iters, train_statistic['p2_acc'], color='blue', label='train_p2_acc')
    # plt.xlabel('epoch')
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig(train_results_img_save_path + 'atda_train_process_' + str(i) +'_.png', bbox_inches='tight')


    plt.figure(figsize=(30, 10))
    plt.subplot(131)
    # plt.plot(iters, train_statistic['p1_loss'], color='red', label='train_p1_loss')
    # plt.plot(iters, val_statistic['p1_loss'], color='darkred', label='val_p1_loss')
    # plt.plot(iters, train_statistic['p2_loss'], color='green', label='train_p2_loss')
    # plt.plot(iters, val_statistic['p2_loss'], color='lightgreen', label='val_p2_loss')
    plt.plot(iters, train_statistic['pt_loss'], color='blue', label='train_pt_loss')
    plt.plot(iters, val_statistic['pt_loss'], color='slateblue', label='val_pt_loss')
    # plt.plot(iters, train_statistic['w_diff_loss'], color='pink', label='weight_diff_loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)


    plt.subplot(132)
    # plt.plot(iters, train_statistic['p1_acc'], color='red', label='train_p1_acc')
    # plt.plot(iters, val_statistic['p1_acc'], color='darkred', label='val_p1_acc')
    # plt.plot(iters, train_statistic['p2_acc'], color='green', label='train_p2_acc')
    # plt.plot(iters, val_statistic['p2_acc'], color='lightgreen', label='val_p2_acc')
    plt.plot(iters, train_statistic['pt_acc'], color='blue', label='train_pt_acc')
    plt.plot(iters, val_statistic['pt_acc'], color='slateblue', label='val_pt_acc')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)


    plt.subplot(133)
    plt.plot(iters, val_statistic['val_f1'], color='red', label='val_f1')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(train_results_img_save_path + 'atda_loss_acc_f1_' + str(i) +'_.png', bbox_inches='tight')




def zero_grad():
    optimizer_F.zero_grad()
    optimizer_Ft.zero_grad()


if __name__ == '__main__':
    since = time.time()
    LEARNING_RATE = args.init_lr
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    EPOCHS = args.EPOCHS
    id = args.id
    logit = args.logit

    # loss_func = utils.focal_loss_zhihu
    loss_func = F.cross_entropy
    print('==use CE loss====')


    drop_prob_list = args.drop_keep
    if args.target == 'DS2' and args.source == 'DS1':
        G = block_network.AlexNetforEcg_DS1_to_DS2().to(device)
    else:
        G = block_network.AlexNetforEcg_DS1_to_DS2().to(device)

    Ft = block_network.EcgClassifier(dropout_keep=0.3, num_classes=N_CLASS).to(device)

    print('F:', G)
    print('Ft:', Ft)

    optimizer_F = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, weight_decay=args.weights_decay)
    optimizer_Ft = torch.optim.Adam(Ft.parameters(), lr=LEARNING_RATE, weight_decay=args.weights_decay)

    optimizer = [optimizer_F, optimizer_Ft]
    model = [G, Ft]
    train_statistic = {'total_loss': [], 'p1_loss': [], 'p2_loss': [], 'pt_loss': [], 'w_diff_loss': [],
                       'p1_acc': [], 'p2_acc': [], 'pt_acc': [],  'new_labeled_samples': []}
    val_statistic = {'total_loss': [], 'p1_loss': [], 'p2_loss': [], 'pt_loss': [], 'w_diff_loss': [],
                       'p1_acc': [], 'p2_acc': [], 'pt_acc': [], 'val_f1': []}

    print('\n==============Training...==================')

    best_val_f1 = 0
    best_train_acc = 0
    best_loss = 1000
    step = 0

    for e in range(0, EPOCHS):

        lr_t = LEARNING_RATE / (1 + 10 * e/EPOCHS) ** 0.75
        w_loss_ratio = args._lambda
        if args.use_decay:
            print('\n==Using lr decay====')
            optimizer_F = torch.optim.Adam(G.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_Ft = torch.optim.Adam(Ft.parameters(), lr=lr_t, weight_decay=args.weights_decay)

        print('\n###current lr:{:.6f}  weight_decay:{:.4f}  lambda:{:.4f}   mix_alpha:{:.2f}  mix:{}  vat:{}  '
              .format(optimizer_F.param_groups[0]['lr'], args.weights_decay, args._lambda, args.mix_alpha, args.mixup, args.VAT))

        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)
        pt_loss, pt_acc = train(x_train, y_train, x_target, model, optimizer, e, w_loss_ratio)

        print('---Train EPOCH: {}/{}, pt_loss: {:.4f}, pt_acc: {:.4f}'
              .format(e + 1, EPOCHS, pt_loss, pt_acc))



        val_pt_loss, val_pt_acc, best_val_f1, current_val_f1, val_f1_cls = val(x_val, y_val, model, e, best_val_f1,
                                                                      w_loss_ratio)
        # train_statistic['total_loss'].append(total_loss)
        # train_statistic['p1_loss'].append(p1_loss)
        # train_statistic['p2_loss'].append(p2_loss)
        train_statistic['pt_loss'].append(pt_loss)
        # train_statistic['w_diff_loss'].append(w_diff_loss)
        # train_statistic['p1_acc'].append(p1_acc)
        # train_statistic['p2_acc'].append(p2_acc)
        train_statistic['pt_acc'].append(pt_acc)


        print('val_pt_loss: {:.4f}, val_pt_acc: {:.4f}\n'.format( val_pt_loss,  val_pt_acc,))

        # val_statistic['total_loss'].append(val_total_loss)
        # val_statistic['p1_loss'].append(val_p1_loss)
        # val_statistic['p2_loss'].append(val_p2_loss)
        val_statistic['pt_loss'].append(val_pt_loss)
        # val_statistic['w_diff_loss'].append(val_w_diff_loss)
        # val_statistic['p1_acc'].append(val_p1_acc)
        # val_statistic['p2_acc'].append(val_p2_acc)
        val_statistic['pt_acc'].append(val_pt_acc)
        val_statistic['val_f1'].append(current_val_f1)

    plot_statistic(train_statistic, val_statistic, id)

    print('\n==============Final Evaluation...==================')

    print('Evaluate target....')
    test(x_test, y_test, model, domain='target_test')
    test(x_target, y_target, model, domain='target_train')
    test(x_train, y_train, model, domain='source_train')
    X_test = torch.cat((x_target, x_test), dim=0)
    Y_test = torch.cat((y_target, y_test), dim=0)
    test(X_test, Y_test, model, domain='target_full')

    if args.TSNE:
        print('=== TSNE plot =====')
        vis(args.id)
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    with open('exp0/' + 'run_' + str(args.id) + '_results.txt', 'a') as f:
        f.write('current_time: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()) + '\n\n')

