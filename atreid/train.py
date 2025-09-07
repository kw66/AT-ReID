from others.utils import *
import torch.nn as nn
from torch.utils.data import DataLoader
from others.transforms import get_transform
from dataset.datamanager import ImageDataset
from dataset.atustc import atustc
from others.optim import get_param_groups, adjust_learning_rate, get_optim
from args.args import create_argparser
from dataset.sample import PSKSampler
from model.resnet import resnet
from loss.ciftloss import TripletLoss
from others.test import attest

args = create_argparser().parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
set_seed(args.seed)
start_time, start = gpu_avaliable(args, memory=args.mb)
checkpoint_path = mkdir_(args)

dataset = atustc()
sampler = PSKSampler(dataset.train, args.p, args.k, args.n, args.sample)
transform_train, transform_test = get_transform(args)
trainloader = DataLoader(
    ImageDataset(dataset.train, transform=transform_train),
    sampler=sampler, batch_size=args.p * args.k,
    num_workers=args.workers, pin_memory=True, drop_last=True,
)

model = resnet(num_p=dataset.num_p)
model.cuda()
param_groups = get_param_groups(args, args.lr, model)
optimizer = get_optim(args, param_groups)

criterion_id = nn.CrossEntropyLoss()
criterion_id.to('cuda')
criterion_tri = TripletLoss()
criterion_tri.to('cuda')


def train(epoch):
    model.train()
    train_loss, data_time, batch_time, grad, acc, \
    loss_1, loss_2, loss_3, loss_4, loss_5, loss_6 = [AverageMeter() for i in range(11)]
    loss_i = [loss_1, loss_2, loss_3, loss_4, loss_5, loss_6]
    loss1 = loss2 = loss3 = loss4 = loss5 = loss6 = torch.Tensor([0]).cuda()
    end = time.time()
    for batch_idx, (imgs, pids, cids, mids, camids, indexs) in enumerate(trainloader):
        optimizer.zero_grad()
        imgs, pids, cids, mids, camids = imgs.cuda(), pids.cuda(), cids.cuda(), mids.cuda(), camids.cuda()
        data_time.update(time.time() - end)
        end = time.time()
        p, y = model(imgs)
        loss1 = criterion_id(y, pids)
        if epoch > args.warmup_loss:
            loss2 = criterion_tri(p, pids)
        _, predicted = y.max(1)
        lossi = [loss1, loss2, loss3, loss4, loss5, loss6]
        loss = sum(lossi)
        loss.backward()
        if args.clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        train_loss.update(loss.item(), pids.size(0))
        for i in range(6):
            loss_i[i].update(lossi[i].item(), pids.size(0))
        grad.update(grad_norm)
        acc.update(100. * predicted.eq(pids).sum().item() / pids.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        t1 = int(time.time() - start)
        step = epoch - 1 + (batch_idx + 1) / len(trainloader)
        t2 = int((time.time() - start) * (args.max_epoch - step) / step)
        if (batch_idx + 1) % max(1, (len(trainloader) // 10)) == 0:
            print(f'E[{epoch:02d}][{batch_idx + 1:3d}/{len(trainloader)}] '
                  f'L: {train_loss.val:.3f} ({train_loss.avg:.3f}) '
                  f'L1: {loss_1.val:.3f} ({loss_1.avg:.3f}) '
                  f'L2: {loss_2.val:.3f} ({loss_2.avg:.3f}) '
                  f'L3: {loss_3.val:.3f} ({loss_3.avg:.3f}) '
                  f'L4: {loss_4.val:.3f} ({loss_4.avg:.3f}) '
                  f'L5: {loss_5.val:.3f} ({loss_5.avg:.3f}) '
                  f'L6: {loss_6.val:.3f} ({loss_6.avg:.3f}) '
                  f'Acc: {acc.val:<4.1f} ({acc.avg:<4.1f}) '
                  f'grad: {grad.val:<4.2f} ({grad.avg:<4.2f}) '
                  f'Time: {t1 // 3600:d}h{t1 // 60 % 60:d}m{t1 % 60:d}s/{t2 // 3600:d}h{t2 // 60 % 60:d}m{t2 % 60:d}s')
    print(f'DataTime: {data_time.sum:.3f} ({data_time.avg:.3f}) BatchTime: {batch_time.sum:.3f} ({batch_time.avg:.3f})')


if test_model(checkpoint_path, model) or args.test:
    print('no training')
    cmc, mAP = attest(args, dataset, model)
    exit()

print('==> Start Training...')
set_seed(args.seed)
best_acc, best_epoch = 0, 0
for epoch in range(1, args.max_epoch + 1):
    current_lr = adjust_learning_rate(args, optimizer, epoch, model)
    print(f'==> Start Training Epoch: {epoch}   lr {current_lr:.6f}   {args.d} v{args.v}   gpu {args.gpu}  {best_acc:.2%}({best_epoch})')
    train(epoch)
    if epoch % args.test_epoch == 0 or epoch >= args.max_epoch - args.last_test:
        print(f'==> Start Testing Epoch: {epoch}')
        #cmc, mAP = np.ones(20), 1.0
        cmc, mAP = attest(args, dataset, model)
        if cmc[0] > best_acc:
            best_acc, best_epoch = cmc[0], epoch
            state = {'model': model.state_dict(), 'epoch': epoch, }
            torch.save(state, checkpoint_path + 'epoch_best.t')

print(f'{cmc[0] * 100:-2.2f}\t{mAP * 100:-2.2f}')
print(f'start:{start_time}\n end :{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
log = f'{args.v:-3d}  Rank-1: {cmc[0]:.2%}  Rank-5: {cmc[4]:.2%}  Rank-10: {cmc[9]:.2%}  ' \
      f'Rank-20: {cmc[19]:.2%}  mAP: {mAP:.2%} \n'
with open('./' + args.d + 'log.txt', 'a') as f:
    f.write(log)
with open(os.path.join(checkpoint_path, 'complete.txt'), 'w') as f:
    f.write(log)

if __name__ == '__main__':
    pass
