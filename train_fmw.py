import argparse, logging, os, datetime, time
import torch.optim
# import matplotlib.pyplot as plt
# from torch.utils.tensorboard import SummaryWriter
from meta import *
from models.resnet import *
from models.vgg import *
from models.densenet import *
from noisy_long_tail_CIFAR import *
from utils import *
from sklearn.cluster import KMeans
from collections import deque

parser = argparse.ArgumentParser(description='First_Order_Meta_Weight_Net')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=333)
parser.add_argument('--x_net_hidden_size', type=int, default=100)
parser.add_argument('--x_net_num_layers', type=int, default=1)
parser.add_argument('--model_type', type=str, default='resnet32')
parser.add_argument('--opt_type', type=str, default='sgd')

parser.add_argument('--lr', type=float, default=.1)
parser.add_argument('--momentum', type=float, default=.9)
parser.add_argument('--dampening', type=float, default=0.)
parser.add_argument('--nesterov', type=bool, default=False)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--meta_weight_decay', type=float, default=1e-4)

parser.add_argument('--save_path', type=str, default='workplace')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--imbalanced_factor', type=int, default=None)
parser.add_argument('--corruption_type', type=str, default=None)
parser.add_argument('--corruption_ratio', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_epoch', type=int, default=120)

parser.add_argument('--meta_interval', type=int, default=20)
parser.add_argument('--print_interval', type=int, default=50)
parser.add_argument('--eta', type=float, default=.5)

args = parser.parse_args()
print(args)

def set_logger(fname):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(fname, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
if args.imbalanced_factor is not None:
    max_epoch = 200
    meta_lr = 1e-4
    meta_weight_decay = 1e-5
    workspace_root = "{}/{}_{}_{}_imb{}/fmw/seed{}_mlr{}_mint{}_eta{}/".format(args.save_path, args.model_type, args.opt_type,
                                                                     args.dataset, args.imbalanced_factor,
                                                                     args.seed, args.meta_lr, args.meta_interval, args.eta)
elif args.corruption_type is not None and args.dataset == 'cifar10':
    max_epoch = 120
    meta_lr = 1e-3
    meta_weight_decay = 1e-4
    workspace_root = "{}/{}_{}_{}_{}_{}/fmw/seed{}_mlr{}_mint{}_eta{}/".format(args.save_path, args.model_type, args.opt_type,
                                                             args.dataset, args.corruption_type, args.corruption_ratio,
                                                             args.seed, args.meta_lr, args.meta_interval, args.eta)
elif args.corruption_type is not None and args.dataset == 'cifar100':
    max_epoch = 150
    meta_lr = 1e-3
    meta_weight_decay = 1e-4
    workspace_root = "{}/{}_{}_{}_{}_{}/fmw/seed{}_mlr{}_mint{}_eta{}/".format(args.save_path, args.model_type,
                                                                           args.opt_type,
                                                                           args.dataset, args.corruption_type,
                                                                           args.corruption_ratio,
                                                                           args.seed, args.meta_lr, args.meta_interval,
                                                                           args.eta)

if not os.path.exists(workspace_root):
    os.makedirs(workspace_root)
    os.makedirs(workspace_root + 'x_net/')
set_logger(os.path.join(workspace_root, 'log.txt'))
logging.info(args)

record = {"p_loss_new": [], "loss": [], "loss_q": [], "loss_test": [], "acc_te": [], 'lmbd': [], 'time': []}

def create_model(model_type):
    if model_type == 'resnet20':
        if args.dataset == 'cifar10':
            y_net = ResNet20(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = ResNet20(num_classes=100).to(device=args.device)
    elif model_type == 'resnet32':
        if args.dataset == 'cifar10':
            y_net = ResNet32(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = ResNet32(num_classes=100).to(device=args.device)
    elif model_type == 'resnet44':
        if args.dataset == 'cifar10':
            y_net = ResNet44(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = ResNet44(num_classes=100).to(device=args.device)
    elif model_type == 'resnet56':
        if args.dataset == 'cifar10':
            y_net = ResNet56(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = ResNet56(num_classes=100).to(device=args.device)
    elif model_type == 'resnet110':
        if args.dataset == 'cifar10':
            y_net = ResNet110(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = ResNet110(num_classes=100).to(device=args.device)
    elif model_type == 'wrn56_2':
        if args.dataset == 'cifar10':
            y_net = WRN56_2(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = WRN56_2(num_classes=100).to(device=args.device)
    elif model_type == 'wrn56_4':
        if args.dataset == 'cifar10':
            y_net = WRN56_4(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = WRN56_4(num_classes=100).to(device=args.device)
    elif model_type == 'wrn56_8':
        if args.dataset == 'cifar10':
            y_net = WRN56_8(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = WRN56_8(num_classes=100).to(device=args.device)

    elif model_type == 'densenet121':
        if args.dataset == 'cifar10':
            y_net = DenseNet121(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = DenseNet121(num_classes=100).to(device=args.device)
    elif model_type == 'densenet161':
        y_net = DenseNet161().to(device=args.device)
    elif model_type == 'densenet169':
        y_net = DenseNet169().to(device=args.device)
    elif model_type == 'densenet201':
        y_net = DenseNet201().to(device=args.device)

    elif model_type == 'vgg9':
        if args.dataset == 'cifar10':
            y_net = VGG9(num_classes=10).to(device=args.device)
        if args.dataset == 'cifar100':
            y_net = VGG9(num_classes=100).to(device=args.device)
    elif model_type == 'vgg16':
        y_net = VGG16().to(device=args.device)
    elif model_type == 'vgg19':
        y_net = VGG19().to(device=args.device)
    else:
        print("*" * 50)
        print("Type Error: The model you choose is not supported!!! Check it!!!")
        print("*" * 50)

    return y_net


def meta_weight_net():
    set_cudnn(device=args.device)
    set_seed(seed=args.seed)

    x_net = MLP(hidden_size=args.x_net_hidden_size, num_layers=args.x_net_num_layers).to(device=args.device)
    y_net = create_model(args.model_type)

    criterion = nn.CrossEntropyLoss().to(device=args.device)

    params_zeros = []
    for param in x_net.parameters():
        params_zeros.append(torch.zeros_like(param).to(device=args.device))
    lmbd = torch.zeros(1)

    if args.opt_type == 'sgd':
        opt_x = torch.optim.SGD(x_net.parameters(), lr=meta_lr, weight_decay=meta_weight_decay)
    elif args.opt_type == 'adam':
        opt_x = torch.optim.Adam(x_net.parameters(), lr=meta_lr, weight_decay=meta_weight_decay)
    else:
        print("*" * 50)
        print("Type Error: The optimizer you choose is not supported!!! Check it!!!")
        print("*" * 50)

    opt_y = torch.optim.SGD(y_net.parameters(), lr=args.lr, momentum=args.momentum, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)
    lr = args.lr
    eta = args.eta

    train_dataloader, meta_dataloader, test_dataloader, imbalanced_num_list = build_dataloader(
        seed=args.seed,
        dataset=args.dataset,
        num_meta_total=args.num_meta,
        imbalanced_factor=args.imbalanced_factor,
        corruption_type=args.corruption_type,
        corruption_ratio=args.corruption_ratio,
        batch_size=args.batch_size,
    )

    meta_dataloader_iter = iter(meta_dataloader)

    if args.dataset == 'cifar10':
        num_class = 10
    elif args.dataset == 'cifar100':
        num_class = 100
    else:
        print("*" * 50)
        print("Type Error: The dataset you choose is not supported!!! Check it!!!")
        print("*" * 50)

    recent_accs = deque(maxlen=5)
    recent_losses = deque(maxlen=5)
    recent_times = deque(maxlen=5)
    print('Training...')
    for epoch in range(max_epoch):
        time1_e = time.time()
        if args.imbalanced_factor is not None:
            if epoch >= 160 and epoch % 20 == 0:
                lr = lr / 100
        elif args.corruption_type is not None and args.dataset == 'cifar10':
            if epoch >= 80 and epoch % 20 == 0:
                lr = lr / 10
        elif args.corruption_type is not None and args.dataset == 'cifar100':
            if epoch >= 80 and epoch % 40 == 0:
                lr = lr / 10
        for group in opt_y.param_groups:
            group['lr'] = lr

        for iteration, (inputs, labels) in enumerate(train_dataloader):
            y_net.train()
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if (iteration + 1) % args.meta_interval == 0:
                p_net = create_model(args.model_type)
                opt_p = torch.optim.SGD(p_net.parameters(), lr=lr)
                p_net.load_state_dict(y_net.state_dict())
                p_net.train()
                p_outputs = p_net(inputs)
                p_loss_vector = F.cross_entropy(p_outputs, labels.long(), reduction='none')
                p_loss_vector_reshape = torch.reshape(p_loss_vector, (-1, 1))
                with torch.no_grad():
                    p_weight = x_net(p_loss_vector_reshape.data)
                p_loss = torch.mean(p_weight * p_loss_vector_reshape)

                opt_p.zero_grad()
                p_loss.backward()
                opt_p.step()

                with torch.no_grad():
                    p_outputs = p_net(inputs)
                    p_loss_vector = F.cross_entropy(p_outputs, labels.long(), reduction='none')
                    p_loss_vector_reshape = torch.reshape(p_loss_vector, (-1, 1))
                p_weight = x_net(p_loss_vector_reshape.data)
                p_loss_new = torch.mean(p_weight * p_loss_vector_reshape)

                outputs = y_net(inputs)
                loss_vector = F.cross_entropy(outputs, labels.long(), reduction='none')
                loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
                weight = x_net(loss_vector_reshape.data)
                loss = torch.mean(weight * loss_vector_reshape)
                loss_q = loss-p_loss_new

                y_net.zero_grad()
                x_net.zero_grad()
                grads2 = torch.autograd.grad(loss_q, x_net.parameters(), only_inputs=True, retain_graph=True)
                grads1 = torch.autograd.grad(loss_q, y_net.parameters(), only_inputs=True)


                try:
                    meta_inputs, meta_labels = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_dataloader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)

                meta_inputs, meta_labels = meta_inputs.to(args.device), meta_labels.to(args.device)
                meta_outputs = y_net(meta_inputs)
                meta_loss = criterion(meta_outputs, meta_labels.long())

                y_net.zero_grad()
                grads_outer = torch.autograd.grad(meta_loss, y_net.parameters(), only_inputs=True)

                dot = 0
                norm_dq = 0

                for df1, dg1 in zip(grads_outer + tuple(params_zeros), grads1 + grads2):
                    df1, dg1 = df1.view(-1), dg1.view(-1)
                    dot += torch.dot(df1, dg1)
                    norm_dq += torch.sum(dg1 ** 2)

                lmbd = F.relu(eta - dot / (norm_dq + 1e-8))

                opt_y.zero_grad()
                for g1, g2, param in zip(grads_outer, grads1, y_net.parameters()):
                    param.grad = g1 + lmbd * g2
                opt_y.step()
                del grads_outer, grads1

                opt_x.zero_grad()
                for g1, param in zip(grads2, x_net.parameters()):
                    param.grad = lmbd * g1
                opt_x.step()
                del grads2

            else:
                outputs = y_net(inputs)
                loss_vector = F.cross_entropy(outputs, labels.long(), reduction='none')
                loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
                with torch.no_grad():
                    weight = x_net(loss_vector_reshape.data)
                loss = torch.mean(weight * loss_vector_reshape)

                opt_y.zero_grad()
                loss.backward()
                opt_y.step()

            if (iteration + 1) % args.print_interval == 0:
                logging.info(
                    "Iter[Epoch]: {}[{}]\tloss_tr: {:.4f}\tloss_q: {:.4f}\tloss_meta: {:.4f}\tlmbd: {:.4f}\tlr: {}".format(
                        iteration + 1, epoch, loss.item(), loss_q.item(), meta_loss.item(), lmbd.item(), lr))
        time2_e = time.time()
        test_loss, test_accuracy = compute_loss_accuracy(
            net=y_net,
            data_loader=test_dataloader,
            criterion=criterion,
            device=args.device,
        )
        recent_accs.append(test_accuracy)
        recent_losses.append(test_loss)
        recent_times.append(time2_e-time1_e)

        logging.info("Epoch: {}\tloss_tr: {:.4f}\tloss_q: {:.4f}\tloss_te: {:.4f}\tacc_te: {:.2%} loss_te_avg: {:.4f}\tacc_te_avg: {:.2%} time_avg: {:.4f} lr: {} ".format(
            epoch, loss.item(), loss_q.item(), test_loss, test_accuracy, sum(recent_losses) / len(recent_losses), sum(recent_accs) / len(recent_accs), sum(recent_times) / len(recent_times), lr))
        if "p_loss_new" in record:
            record["p_loss_new"].append((epoch, p_loss_new.item()))
        if "loss" in record:
            record["loss"].append((epoch, loss.item()))
        if "loss_q" in record:
            record["loss_q"].append((epoch, loss_q.item()))
        if "loss_test" in record:
            record["loss_test"].append((epoch, test_loss))
        if "acc_te" in record:
            record["acc_te"].append((epoch, test_accuracy))
        if "lmbd" in record:
            record["lmbd"].append((epoch, lmbd.item()))
        if "time" in record:
            record["time"].append((epoch, time2_e-time1_e))
        if (epoch + 1) % 20 == 0:
            torch.save(x_net.state_dict(), os.path.join(workspace_root, "x_net/x_net_{}.pth".format(epoch)))
    torch.save(record, os.path.join(workspace_root, "record.pt"))
    torch.save(y_net.state_dict(), os.path.join(workspace_root, "net.pth"))
    torch.save(x_net.state_dict(), os.path.join(workspace_root, "x_net.pth"))



if __name__ == '__main__':
    meta_weight_net()
