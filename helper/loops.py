from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy
import wandb
import matplotlib.pyplot as plt


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    losses_cls = AverageMeter()
    losses_kd = AverageMeter()
    losses_div = AverageMeter() if opt.alpha > 0 else None
    losses_hkd = AverageMeter() if opt.hkd_weight > 0 else None
    losses_ot = AverageMeter() if opt.ot_weight > 0 else None


    top1 = AverageMeter()
    top5 = AverageMeter()

    M = None
    P = None

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd', 'hkd', 'ceot']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd', 'hkd', 'ceot']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        # Initialize loss_kd as a zero tensor
        loss_kd = torch.tensor(0.0, device=input.device)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = torch.tensor(0.0, device=input.device)
        elif opt.distill == 'ot':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd, P, M = criterion_kd(f_s, f_t)
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'hkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(epoch, f_s, logit_s, f_t, logit_t, index, contrast_idx)
        elif opt.distill == 'ceot':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd, loss_hkd, loss_ot, P, M = criterion_kd(epoch, f_s, logit_s, f_t, logit_t, index, contrast_idx)
            if losses_hkd is not None:
                losses_hkd.update(loss_hkd.item(), input.size(0))
            if losses_ot is not None:
                losses_ot.update(loss_ot.item(), input.size(0))
        elif opt.distill == 'gnnot':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd, P, M = criterion_kd(epoch, f_s, logit_s, f_t, logit_t)
        elif opt.distill == 'gnngw':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd, P, M = criterion_kd(epoch, f_s, logit_s, f_t, logit_t)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = torch.tensor(0.0, device=input.device)
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = torch.tensor(0.0, device=input.device)
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        losses_cls.update(loss_cls.item(), input.size(0))
        losses_kd.update(loss_kd.item(), input.size(0))

        if losses_div is not None:
            losses_div.update(loss_div.item(), input.size(0))

        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    if (epoch % 5 == 0 or epoch == 1) and M is not None and P is not None:

        plt.figure()
        plt.imshow(M.detach().cpu().numpy(), cmap='hot')
        plt.title(f"M Matrix Heatmap at Epoch {epoch}")
        plt.colorbar()
        plt.savefig('M_heatmap.png')
        plt.close()

        plt.figure()
        plt.imshow(P.detach().cpu().numpy(), cmap='hot')
        plt.title(f"P Matrix Heatmap at Epoch {epoch}")
        plt.colorbar()
        plt.savefig('P_heatmap.png')
        plt.close()
        # 记录热图到WandB，不手动设置步骤数
        wandb.log({
            "M Matrix Heatmap": wandb.Image('M_heatmap.png', caption=f"M Matrix Epoch {epoch}"),
            "P Matrix Heatmap": wandb.Image('P_heatmap.png', caption=f"P Matrix Epoch {epoch}")
        })

    log_data = {
        "Epoch": epoch,
        "Train/Loss": losses.avg,
        "Train/Loss_Cls": losses_cls.avg,
        "Train/Loss_KD": losses_kd.avg,
        "Train/Acc@1": top1.avg,
        "Train/Acc@5": top5.avg,

    }

    if losses_div is not None:
        log_data["Train/Loss_Div"] = losses_div.avg
    if losses_hkd is not None:
        log_data["Train/Loss_hkd"] = losses_hkd.avg
    if losses_ot is not None:
        log_data["Train/Loss_ot"] = losses_ot.avg

    wandb.log(log_data)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
        log_data = {
            "Test/Loss": losses.avg,
            "Test/Acc@1": top1.avg,
            "Test/Acc@5": top5.avg
        }
        wandb.log(log_data)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
