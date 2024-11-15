"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import Embed, ConvReg, LinearEmbed, SelfA
from models.util import Connector, Translator, Paraphraser

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.tinyimagenet import get_tiny_imagenet_dataloaders, get_tiny_imagenet_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader, get_dataloader_sample

from helper.util import adjust_learning_rate

from distiller_zoo.KD import DistillKL
from distiller_zoo.OTGD import OTGDLoss
from distiller_zoo import Attention, Similarity, Correlation, VIDLoss, RKDLoss, GNNLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, FSP, IRG, HKDOTLoss
from distiller_zoo import SemCKDLoss
from crd.criterion import CRDLoss

from helper.loops_ot import train_distill as train, validate
from helper.pretrain import init

import wandb


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    parser.add_argument('--wandb_name', type=str, default='default_name', help='Name of the W&B run')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'tiny_imagenet', 'imagenet'],
                        help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd', 'fsp', 'semckd',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'hkd', 'irg',
                                                                      'ceot', 'otgd'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # OT para for node loss
    parser.add_argument('--ot_gamma', type=float, default=1, help='strength of entropy regularization')
    parser.add_argument('--ot_eps', type=float, default=1e-5, help='control the stopping condition for iterations')
    parser.add_argument('--ot_iter', type=int, default=2, help='the maximum number of iterations')
    parser.add_argument('--ot_reg', type=float, default=0, help='the maximum number of iterations')
    parser.add_argument('--ot_method', type=str, default='pcc', choices=['pcc', 'cos', 'edu'])
    parser.add_argument('--M_norm', type=str, default='Mz', choices=['Mz', 'Mm', 'Mmz', None])
    parser.add_argument('--P_norm', type=str, default='Prc', choices=['Pr', 'Pc', 'Prc',  None])
    parser.add_argument('--device', type=str, default='cuda', help='')

    # OT HKD
    parser.add_argument('--hkd_weight', type=float, default=1, help='weight for hkd')
    parser.add_argument('--ot_weight', type=float, default=1, help='weight for ot')
    parser.add_argument('--e_weight', type=float, default=1, help=' ')
    parser.add_argument('--g_weight', type=float, default=1, help=' ')
    parser.add_argument('--ge_weight', type=float, default=1, help=' ')
    parser.add_argument('--eg_weight', type=float, default=1, help=' ')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax', 'hkd'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # IRG distillation
    parser.add_argument('--w_irg_vert', type=float, default=0.1, help='weight for IRG vertex')
    parser.add_argument('--w_irg_edge', type=float, default=5.0, help='weight for IRG edge')
    parser.add_argument('--w_irg_tran', type=float, default=5.0, help='weight for IRG transformation')
    parser.add_argument('--transform_layer_t', nargs='+', type=int, default=[])
    parser.add_argument('--transform_layer_s', nargs='+', type=int, default=[])
    parser.add_argument('--no_edge_transform', action='store_true')  # default=false

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    if torch.cuda.is_available():
        map_location = 'cuda'
    else:
        map_location = 'cpu'
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    print('==> done')
    return model


def main():
    wandb.login(key='4d34516b8aa63d2e12c5116f75a0fcf604fe6a57')
    opt = parse_option()  # 存储程序的参数和配置
    wandb.init(project="HKD", name=opt.wandb_name, save_code=True)
    wandb.run.log_code(root="distiller_zoo/")

    best_acc = 0
    best_acc_top5 = 0

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd', 'hkd', 'ceot', 'hkdot']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True)
    elif opt.dataset == 'tiny_imagenet':
        if opt.distill in ['crd', 'hkd', 'ceot']:
            train_loader, val_loader, n_data = get_tiny_imagenet_dataloaders_sample(batch_size=opt.batch_size,
                                                                                    num_workers=opt.num_workers,
                                                                                    k=opt.nce_k,
                                                                                    mode=opt.mode)
        else:
            train_loader, val_loader, n_data = get_tiny_imagenet_dataloaders(batch_size=opt.batch_size,
                                                                             num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.distill in ['crd', 'hkd']:
            train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset,
                                                                                       batch_size=opt.batch_size,
                                                                                       num_workers=opt.num_workers,
                                                                                       is_sample=True, k=opt.nce_k)
        else:
                train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                                  num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)

    class_num_map = {
        'cifar100': 100,
        'tiny_imagenet': 200
    }
    if opt.dataset not in class_num_map:
        raise NotImplementedError(opt.dataset)
    n_cls = class_num_map[opt.dataset]

    # model
    print('student network is:', opt.model_s)
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)

    print('distill loss is:', opt.distill, '\n')

    if opt.distill in ['ot', 'ceot', 'otgd']:
        print('cost matrix method is: ', opt.ot_method, '\n',
              '----- OT gamma is ', opt.ot_gamma, ', eps is ', opt.ot_eps, ', max_iter is ', opt.ot_iter, '-----')
    if opt.distill in ['ceot']:
        print('\n', 'hkd weight is: ', opt.hkd_weight, ' ot weight is: ', opt.ot_weight)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'ceot':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = HKDOTLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        module_list.append(criterion_kd.gnn_s)
        module_list.append(criterion_kd.gnn_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.gnn_s)
        trainable_list.append(criterion_kd.gnn_t)
    elif opt.distill == 'otgd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        criterion_kd = OTGDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        module_list.append(criterion_kd.gnn_s)
        module_list.append(criterion_kd.gnn_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.gnn_s)
        trainable_list.append(criterion_kd.gnn_t)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'semckd':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(len(feat_s) - 2, len(feat_t) - 2, opt.batch_size, s_n, t_n)
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    elif opt.distill == 'hkd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = GNNLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        module_list.append(criterion_kd.gnn_s)
        module_list.append(criterion_kd.gnn_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.gnn_s)
        trainable_list.append(criterion_kd.gnn_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'irg':
        criterion_kd = IRG()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_top5 = test_acc_top5
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
                'best_acc_top5': best_acc_top5,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch. 
    print('best accuracy:', best_acc)
    print('top-5 accuracy:', best_acc_top5)


    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
