import argparse
import numpy as np
import random
import scipy.io as sio
import scipy.signal as ss
import os
import os.path as osp
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn import manifold

# np.random.seed(666)

class FeatureVisualize(object):
    '''
    Visualize features by TSNE
    '''

    def __init__(self, features, labels):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.labels = labels

    def plot_tsne(self, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)
        del features
        for i in range(data.shape[0]):
            plt.text(data[i, 0], data[i, 1], str(self.labels[i]),
                     color=plt.cm.Set1(self.labels[i] / 10.),
                     fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.title('T-SNE')
        if save_eps:
            plt.savefig('tsne.eps', dpi=600, format='eps')
        plt.show()


def vis(exp_id):

    train_results_save_path = '../exp1/train_results/'
    train_results_img_save_path = '../exp1/train_results/imgs/'
    colors_s = {0: 'red', 1: 'green', 2: 'blue', 3: 'darkviolet', 4:'yellow'}
    colors_t = {0: 'lightcoral', 1: 'limegreen', 2: 'cornflowerblue', 3: 'plum', 4:'gold'}
    categories = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
    metrics = {'Cosine': 'cosine', 'L2': 'euclidean'}

    # a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'source_tarin' + '_fea.npz')
    # features_source = a['feature']
    # labels_source = a['label'].squeeze(axis=1).astype(np.int32)
    #
    # a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'target_train' + '_fea.npz')
    # features_target_train = a['feature']
    # labels_target_train = a['label'].squeeze(axis=1).astype(np.int32)
    # features_target = features_target_train
    # labels_target = labels_target_train
    # a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'target_test' + '_fea.npz')
    # features_target_test = a['feature']
    # labels_target_test = a['label'].squeeze(axis=1).astype(np.int32)
    a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'source_train' + '_fea.npz')
    features_source = a['feature']
    labels_source = a['label'].squeeze(axis=1).astype(np.int32)

    # a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'target_train' + '_fea.npz')
    # features_target_train = a['feature']
    # labels_target_train = a['label'].squeeze(axis=1).astype(np.int32)

    # a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'target_test' + '_fea.npz')
    a = np.load(train_results_save_path + 'exp_' + str(exp_id) + "_" + 'target_full' + '_fea.npz')
    features_target_test = a['feature']
    labels_target_test = a['label'].squeeze(axis=1).astype(np.int32)
    
    # features_target = np.concatenate((features_target_train, features_target_test), 0)
    # labels_target = np.concatenate((labels_target_train, labels_target_test), 0)
    features_target = features_target_test
    labels_target = labels_target_test

    print('features_source shape:', features_source.shape)
    print('labels_source shape:', labels_source.shape)
    print('features_target shape:', features_target.shape)
    print('labels_target shape:', labels_target.shape)


    features_s_dict = {}
    labels_s_dict = {}
    features_t_dict = {}
    labels_t_dict = {}
    for l in range(4):
        l_indices = np.argwhere(labels_source == l).squeeze(axis=1)
        features_s_dict[l] = features_source[l_indices]
        labels_s_dict[l] = labels_source[l_indices]

        l_indices_t = np.argwhere(labels_target == l).squeeze(axis=1)
        features_t_dict[l] = features_target[l_indices_t]
        labels_t_dict[l] = labels_target[l_indices_t]

    ''' 5000 samples for N'''
    max_num = args.max_num
    for i in range(4):
        if max(len(features_s_dict[i]), len(features_t_dict[i])) > max_num:
            features_s_dict[i] = features_s_dict[i][:max_num]
            labels_s_dict[i] = labels_s_dict[i][:max_num ]
            features_t_dict[i] = features_t_dict[i][:max_num ]
            labels_t_dict[i] = labels_t_dict[i][:max_num ]

    features_source = np.concatenate((features_s_dict[0], features_s_dict[1], features_s_dict[2],
                                      features_s_dict[3]), 0)
    labels_source = np.concatenate((labels_s_dict[0], labels_s_dict[1], labels_s_dict[2],
                                    labels_s_dict[3]), 0)
    features_target = np.concatenate((features_t_dict[0], features_t_dict[1], features_t_dict[2],
                                      features_t_dict[3]), 0)
    labels_target = np.concatenate((labels_t_dict[0], labels_t_dict[1], labels_t_dict[2],
                                    labels_t_dict[3]), 0)

    # index = np.random.permutation(features_source.shape[0])
    # features_source = features_source[index]
    # labels_source = labels_source[index]
    # index = np.random.permutation(features_target.shape[0])
    # features_target = features_target[index]
    # labels_target = labels_target[index]

    print('features_source shape:', features_source.shape)
    print('labels_source shape:', labels_source.shape)
    print('features_target shape:', features_target.shape)
    print('labels_target shape:', labels_target.shape)
    '''1.t-SNE算法是随机的，使用不同种子进行多次重启可以产生不同的嵌入。 但是，选择嵌入的错误最少是完全合法的。
       2.确保对所有特征使用相同的刻度。因为流形学习方法是基于最近邻搜索的，否则算法可能表现不佳
       3.在一个高维空间中取得一组点，并在一个低维空间(通常是2D 平面)中找到这些点的准确表示。
           该算法是非线性的，适用于底层数据，对不同的区域进行不同的变换
       4.困惑度应该小于样本的数目
       5.如果您看到t-SNE图上有奇怪的“挤压”形状，则该过程可能停止得太早'''
    tsne = manifold.TSNE(n_components=2, metric=args.dist,
                         # perplexity=args.plex,#'cosine'、'euclidean',50
                         perplexity=args.plex,#50,300,500 该参数不重要,一般exag值越大，嵌入空间中自然簇之间的空间越大
                         # early_exaggeration=12.0,
                         early_exaggeration=12,
                         learning_rate=200,  ##200.0,[10,1000]
                         # 如果学习率太高，数据可能看起来像一个“球”，任何点与其最近的邻居的距离都大致相等;
                         # 如果学习率太低，大多数点可能看起来像压缩在密集的云中，没有异常值
                         n_iter=args.iter,  ##2000,>=250
                         init='pca',# PCA初始化通常比随机初始化更全局稳定。
                         # random_state=args.seeds  #666
                         # random_state=666
                         )

    # tsne = TSNE(n_jobs=8, random_state=666)

    count_source = {'N': 0, 'S': 0, 'V': 0, 'F': 0, 'Q':0}
    count_target = {'N': 0, 'S': 0, 'V': 0, 'F': 0, 'Q':0}
    keys = ['N', 'S', 'V', 'F', 'Q']

    num_source = len(labels_source)
    num_target = len(labels_target)


    for i in range(num_source):
        count_source[keys[labels_source[i]]] += 1
    for j in range(num_target):
        count_target[keys[labels_target[j]]] += 1

    for k in keys:
        print('The number of {} in source: {}; in target: {}'.format(k, count_source[k], count_target[k]))



    features = np.concatenate([features_source, features_target], axis=0)
    feat_tsne = tsne.fit_transform(features)
    print('feat_tsne shape:', feat_tsne.shape)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # feat_norm  = scaler.fit_transform(feat_tsne)
    x_min, x_max = feat_tsne.min(0), feat_tsne.max(0)
    feat_norm = (feat_tsne - x_min) / (x_max - x_min)

    # feat_norm = feat_tsne

    feat_norm_s = feat_norm[0: num_source]
    feat_norm_t = feat_norm[num_source: num_target + num_source]

    features_s_dict = {}
    feat_norm_s_dict = {}
    features_t_dict = {}
    feat_norm_t_dict = {}

    for l in range(4):
        l_indices = np.argwhere(labels_source == l).squeeze(axis=1)
        features_s_dict[l] = features_source[l_indices]
        feat_norm_s_dict[l] = feat_norm_s[l_indices]

        l_indices_t = np.argwhere(labels_target == l).squeeze(axis=1)
        features_t_dict[l] = features_target[l_indices_t]
        feat_norm_t_dict[l] = feat_norm_t[l_indices_t]



    '''The feature visualization'''
    # plt.figure(figsize=(30, 15))
    # for i in range(features_source.shape[0]):
    #     if labels_source[i] == 0:
    #         plt.subplot(411)
    #         plt.plot(features_source[i], color=colors_s[labels_source[i]])
    #     elif labels_source[i] == 1:
    #         plt.subplot(412)
    #         plt.plot(features_source[i], color=colors_s[labels_source[i]])
    #     elif labels_source[i] == 2:
    #         plt.subplot(413)
    #         plt.plot(features_source[i], color=colors_s[labels_source[i]])
    #     elif labels_source[i] == 3:
    #         plt.subplot(414)
    #         plt.plot(features_source[i], color=colors_s[labels_source[i]])
    # img_save_path = osp.join(train_results_img_save_path, 'feat_s_{}.png'.format(exp_id))
    # plt.savefig(img_save_path, bbox_inches='tight')
    # plt.close()
    #
    # plt.figure(figsize=(30, 15))
    # for i in range(features_target.shape[0]):
    #     if labels_target[i] == 0:
    #         plt.subplot(411)
    #         plt.plot(features_target[i], color=colors_s[labels_target[i]])
    #     elif labels_target[i] == 1:
    #         plt.subplot(412)
    #         plt.plot(features_target[i], color=colors_s[labels_target[i]])
    #     elif labels_target[i] == 2:
    #         plt.subplot(413)
    #         plt.plot(features_target[i], color=colors_s[labels_target[i]])
    #     elif labels_target[i] == 3:
    #         plt.subplot(414)
    #         plt.plot(features_target[i], color=colors_s[labels_target[i]])
    #
    # img_save_path = osp.join(train_results_img_save_path, 'feat_t_{}.png'.format(exp_id))
    # plt.savefig(img_save_path, bbox_inches='tight')
    # plt.close()

    '''The class-specific view'''
    plt.figure(figsize=(20, 20))
    # plt.figure(figsize=(8, 8))
    for l in range(4):
        plt.scatter(feat_norm_s_dict[l][:, 0], feat_norm_s_dict[l][:, 1],
                    marker='.', color=colors_s[l], label='source {}'.format(categories[l]))
        plt.scatter(feat_norm_t_dict[l][:, 0], feat_norm_t_dict[l][:, 1],
                    marker='x', color=colors_t[l], label='target {}'.format(categories[l]))

    # plt.scatter(feat_norm_s_dict[0][:, 0], feat_norm_s_dict[0][:, 1],
    #             marker='$0$', color='red', label='source {}'.format(categories[0]))
    # plt.scatter(feat_norm_t_dict[0][:, 0], feat_norm_t_dict[0][:, 1],
    #             marker='$0$', color='blue', label='target {}'.format(categories[0]))
    #
    # plt.scatter(feat_norm_s_dict[1][:, 0], feat_norm_s_dict[1][:, 1],
    #             marker='$1$', color='red', label='source {}'.format(categories[1]))
    # plt.scatter(feat_norm_t_dict[1][:, 0], feat_norm_t_dict[1][:, 1],
    #             marker='$1$', color='blue', label='target {}'.format(categories[1]))
    #
    # plt.scatter(feat_norm_s_dict[2][:, 0], feat_norm_s_dict[2][:, 1],
    #             marker='$2$', color='red', label='source {}'.format(categories[2]))
    # plt.scatter(feat_norm_t_dict[2][:, 0], feat_norm_t_dict[2][:, 1],
    #             marker='$2$', color='blue', label='target {}'.format(categories[2]))
    #
    # plt.scatter(feat_norm_s_dict[3][:, 0], feat_norm_s_dict[3][:, 1],
    #             marker='$3$', color='red', label='source {}'.format(categories[3]))
    # plt.scatter(feat_norm_t_dict[3][:, 0], feat_norm_t_dict[3][:, 1],
    #             marker='$3$', color='blue', label='target {}'.format(categories[3]))
    #
    # plt.scatter(feat_norm_s_dict[4][:, 0], feat_norm_s_dict[4][:, 1],
    #             marker='$4$', color='red', label='source {}'.format(categories[4]))
    # plt.scatter(feat_norm_t_dict[4][:, 0], feat_norm_t_dict[4][:, 1],
    #             marker='$4$', color='blue', label='target {}'.format(categories[4]))

    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper right')
    # plt.legend(loc='lower left')
    # plt.legend(loc='lower right')
    img_save_path = osp.join(train_results_img_save_path, 'tsne_{}_cls.png'.format(exp_id))
    plt.savefig(img_save_path, bbox_inches='tight',dpi=300,pad_inches=0)#dpi=600,
    plt.close()

    '''The domain-specific view'''
    # plt.figure(figsize=(20, 20))
    # plt.figure(figsize=(8, 8))
    # for i in range(feat_norm.shape[0]):
    #     if i < num_source:
    #         plt.scatter(feat_norm[i, 0], feat_norm[i, 1],
    #                     marker='.', color='blue')
    #     else:
    #         plt.scatter(feat_norm[i, 0], feat_norm[i, 1],
    #                     marker='x', color='red')
    # plt.xticks([])
    # plt.yticks([])
    # img_save_path = osp.join(train_results_img_save_path, 'tsne_{}_dom.png'.format(exp_id))
    # plt.savefig(img_save_path, bbox_inches='tight')
    # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", dest="exp_id", type=int, default=0)
    parser.add_argument("--seed", dest="seeds", type=int, default=666)
    parser.add_argument("--plex", dest="plex", type=int, default=500)
    parser.add_argument("--iter", dest="iter", type=int, default=2000)
    parser.add_argument("--dist", dest="dist", type=str, default='euclidean')
    parser.add_argument("--max", dest="max_num", type=int, default=1000)
    parser.add_argument("--lr", dest="lr", type=int, default=200)
    args = parser.parse_args()
    print("--plex:{}  --max_num:{}  --dist:{}  --iter:{}".format(args.plex, args.max_num, args.dist, args.iter))
    vis(args.exp_id)



    # a = np.load('../exp1/train_results/' + 'exp_' + str(0) + "_pseudo_acc.npz" )
    # pseudo_acc_0 = a['acc']
    # print(pseudo_acc_0)
    # a = np.load('../exp1/train_results/' + 'exp_' + str(3) + "_labeled_samples.npz")
    # labeled_samples = a['num']
    # a = np.load('../exp1/train_results/' + 'exp_' + str(3) + "_pseudo_acc.npz")
    # pseudo_acc_3 = a['acc']
    # print(pseudo_acc_3)
    # train_results_img_save_path = '../exp1/train_results/imgs/'
    # iters = [i + 1 for i in range(0, len(pseudo_acc_0 ))]
    # plt.figure(figsize=(10, 10))
    # plt.plot(iters, pseudo_acc_0, color='green', label='pseudo_acc_atda')
    # plt.plot(iters, pseudo_acc_3, color='blue', label='pseudo_acc_atda+mixup')
    # plt.xlabel('epoch')
    # plt.legend(loc="best")
    # plt.grid(True)
    # plt.savefig(train_results_img_save_path + 'pseudo_acc_mixup_DS1_DS2' + '_.png', bbox_inches='tight')
    #
