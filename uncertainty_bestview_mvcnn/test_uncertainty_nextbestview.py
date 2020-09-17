import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
#from torchsummary import summary

import numpy as np
import time
import os
import argparse
from scipy.stats import mode

from models.MVCNN import *
from models.RESNET import *
from utils.util import logEpoch, save_checkpoint
from utils.custom_dataset import MultiViewDataset
from utils.my_MNet40_data import myMVMnet40
from utils.logger import Logger

MVCNN = 'mvcnn'
RESNET = 'resnet'
MODELS = [MVCNN, RESNET]

parser = argparse.ArgumentParser(description='MVCNN-PyTorch')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--threshold', default=0.5, type=float, metavar='Th',
                    help='threshold (default: 0.5)')
parser.add_argument('--estimation', '-e', metavar='uncertainty_method', default='softmax', type=str,
                    help='uncertainty estimation methods softmax mcdropout_mean mcdropout_mode')
parser.add_argument('--model', '-m', metavar='MODEL', default=MVCNN, choices=MODELS,
                    help='pretrained model: ' + ' | '.join(MODELS) + ' (default: {})'.format(MVCNN))
parser.add_argument('--depth', choices=[18, 34, 50, 101, 152], type=int, metavar='N', default=18, help='resnet depth (default: resnet18)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run (default: 100)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.0001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--lr-decay-freq', default=30, type=float,
                    metavar='W', help='learning rate decay (default: 30)')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='W', help='learning rate decay (default: 0.1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')

args = parser.parse_args()

print('Loading Multi-view data')

transform = transforms.Compose([
    #transforms.CenterCrop(500),
    #transforms.Resize(224),
    transforms.ToTensor(),
])

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dset_train = myMVMnet40(args.data, 'train', transform=transform)
train_set, val_set = torch.utils.data.random_split(dset_train, [8500, 1343])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

dset_test = myMVMnet40(args.data, 'test', transform=transform)
test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=True, num_workers=2)
test_reject_loader = DataLoader(dset_test, batch_size=1, shuffle=True, num_workers=2)
classes = dset_train.classes
nb_classes = len(classes)
print(len(classes), classes)
print(len(val_loader.dataset))

if args.model == RESNET:
    if args.depth == 18:
        model = resnet18(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 34:
        model = resnet34(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 50:
        model = resnet50(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 101:
        model = resnet101(pretrained=args.pretrained, num_classes=len(classes))
    elif args.depth == 152:
        model = resnet152(pretrained=args.pretrained, num_classes=len(classes))
    else:
        raise Exception('Specify number of layers for resnet in command line. --resnet N')
    print('Using ' + args.model + str(args.depth))
else:
    model = mvcnn(pretrained=args.pretrained,num_classes=len(classes))
    print('Using ' + args.model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
#model = nn.DataParallel(model)
model.to(device)
#print(summary(model, (12, 3, 224, 224)))
cudnn.benchmark = True

#logger = Logger('logs')
# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
criterion2 = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_acc = 0.0
best_loss = 0.0
start_epoch = 0
n_views = 12
thre = args.threshold

# Helper functions
def load_checkpoint():
    global best_acc, start_epoch
    # Load checkpoint.
    print('\n==> Loading checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint file found!'

    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Validation and Testing
def eval(data_loader):
    # Eval
    total = 0.0
    correct = 0.0
    correct_single = 0.0
    correct_view = 0.0
    correct_view2 = 0.0
    correct_view3 = 0.0
    correct_view4 = 0.0
    correct_view5 = 0.0

    total_loss = 0.0
    total_m_loss = 0.0
    total_s_loss = 0.0
    total_view_loss = 0.0
    n = 0
    predictions = []
    target_class = []
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs = torch.from_numpy(inputs)
            #inputs = inputs[:, :6, :, :, :]
            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            m_outputs, outputs_pool, best_view = model(inputs)
            m_loss = criterion(m_outputs, targets)
            s_loss = 0.0
            loss_list = torch.zeros(len(targets), n_views)
            k = 0
            for v in outputs_pool:
                loss_list[:, k] = criterion2(v, targets)
                k += 1
                s_loss += criterion(v, targets)
            idx = np.argsort(loss_list.cpu().numpy(), axis=1)
            smallst = torch.argmin(loss_list, dim=1)
            fake_target = Variable(torch.argmin(loss_list, dim=1).cuda(device))
            fake_target2 = Variable(torch.from_numpy(idx[:, 1]).cuda(device))
            fake_target3 = Variable(torch.from_numpy(idx[:, 2]).cuda(device))
            fake_target4 = Variable(torch.from_numpy(idx[:, 3]).cuda(device))
            fake_target5 = Variable(torch.from_numpy(idx[:, 4]).cuda(device))
            view_loss = 0.0
            for view in best_view:
                view_loss += criterion(view, fake_target)
                _, predicted_view = torch.max(view.data, 1)
                correct_view += (predicted_view.cpu() == fake_target.cpu()).sum()
                correct_view2 += (predicted_view.cpu() == fake_target2.cpu()).sum()
                correct_view3 += (predicted_view.cpu() == fake_target3.cpu()).sum()
                correct_view4 += (predicted_view.cpu() == fake_target4.cpu()).sum()
                correct_view5 += (predicted_view.cpu() == fake_target5.cpu()).sum()

            loss = m_loss + s_loss + view_loss
            total_m_loss += m_loss
            total_s_loss += s_loss
            total_view_loss += view_loss
            total_loss += loss
            n += 1

            # multi-view results
            _, predicted = torch.max(m_outputs.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()
            # single-view results
            for v in outputs_pool:
                _, predicted_s = torch.max(v.data, 1)
                correct_single += (predicted_s.cpu() == targets.cpu()).sum()

            # ROC cruve
            predictions.extend(predicted.cpu().numpy())
            target_class.extend(targets.cpu().numpy())

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n
    avg_m_loss = total_m_loss / n
    avg_s_loss = total_s_loss / n
    avg_view_loss = total_view_loss / n
    avg_view_acc = 100 * correct_view / (total * len(outputs_pool))
    avg_view_acc3 = 100 * (correct_view+correct_view2+correct_view3) / (total * len(outputs_pool))
    avg_test_acc_single = 100 * correct_single / (total * len(outputs_pool))

    return avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_view_acc, avg_view_acc3, avg_view_loss


def eval4bestview(data_loader, is_test=False):

    # Eval
    total = 0.0
    correct = 0.0
    correct_single = 0.0

    total_loss = 0.0
    n = 0
    predictions = []
    target_class = []
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)

            inputs_all = torch.from_numpy(inputs)
            inputs_2 = inputs_all[:, 0, :, :, :]
            inputs, targets = inputs_2.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            m_outputs, outputs_pool, best_view = model(inputs)
            n += 1
            view_list = []
            #for view in best_view:
            #    preds = np.argmax(view.cpu().numpy(), axis=1)
            #    view_list.extend(preds)
            #pred, _ = mode(view_list)
            idx = np.argsort(best_view.cpu().numpy()).flatten()
            view_number = np.argmax(best_view.cpu().numpy())
            inputs_best = inputs_all[:, view_number, :, :, :].unsqueeze(0)
            inputs_top2 = inputs_all[:, idx[1], :, :, :].unsqueeze(0)
            inputs = torch.cat((inputs_top2, inputs_best), dim=1)
            inputs = inputs.cuda(device)
            inputs = Variable(inputs)
            yhats, yhats_s, best_view = model(inputs)
            loss = criterion(yhats, targets)
            total_loss += loss
            # best view results
            _, predicted = torch.max(yhats.data, 1)
            total += targets.size(0)
            correct += (predicted.cpu() == targets.cpu()).sum()

    avg_test_acc = 100 * correct / total
    avg_loss = total_loss / n

    return avg_test_acc, avg_loss


def val_acqu(data_loader):
    # Eval using MC dropout
    model.train()
    predictions = []
    dropout_iterations = 100
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs = torch.from_numpy(inputs)
            #inputs = inputs[:, :6, :, :, :]
            inputs, targets = inputs.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)


            pool_size = len(val_loader.dataset)

            all_correct = np.zeros(shape=(len(inputs)))
            for d in range(dropout_iterations):
                # compute output
                yhats, yhats_s = model(inputs)
                preds = F.softmax(yhats.cpu())
                preds = np.argmax(preds.data.numpy(), axis=-1)
                correct_class = (preds == targets.cpu().numpy())
                correct_class = np.array(correct_class.astype(int))
                all_correct += correct_class

            all_correct = np.divide(all_correct, dropout_iterations)
            all_correct = all_correct.tolist()
            predictions.extend(all_correct)

    confidence_score =  sum(predictions) / float(len(predictions))
    return confidence_score

def softmax_estimation(data_loader):
    total = 0.0
    correct = 0.0
    view_num = 0.0
    views = []
    softmax_list = []
    pred_list = []
    real_list = []
    for i, (inputs, targets)  in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs_all = torch.from_numpy(inputs)
            inputs_2 = inputs_all[:, 0, :, :, :]
            inputs, targets = inputs_2.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            m_outputs, outputs_pool, best_view = model(inputs)
            idx = np.argsort(best_view.cpu().numpy()).flatten()
            view_number = np.argmax(best_view.cpu().numpy())
            inputs_best = inputs_all[:, view_number, :, :, :].unsqueeze(0)

            for v in range(2,n_views+1):
                inputs_topk = inputs_all[:, idx[:v-1], :, :, :]
                inputs = torch.cat((inputs_topk, inputs_best), dim=1)
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                inputs, targets = Variable(inputs), Variable(targets)
                # compute output
                yhats, yhats_s, _ = model(inputs)
                preds = F.softmax(yhats.cpu())
                softmaxs_1 = np.max(preds.data.numpy(), axis=-1)
                if(softmaxs_1 >= thre):
                    break


            _, predicted = torch.max(yhats.data, 1)
            correct += (predicted.cpu() == targets.cpu()).sum()
            view_num += v
            views.append(float(v))
            total += 1
            softmaxs_1 = softmaxs_1.tolist()
            softmax_list.extend(softmaxs_1)
            pred_list.extend(predicted.cpu().numpy())
            real_list.extend(targets.cpu().numpy())

    acc_coverage = 100 * correct / total
    mean_view = view_num / total
    return mean_view, acc_coverage, views, softmax_list, pred_list, real_list



def softmax_estimation2(data_loader):
    total = 0.0
    correct = 0.0
    view_num = 0.0
    views = []
    softmax_list = []
    pred_list = []
    real_list = []
    for i, (inputs, targets)  in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs_all = torch.from_numpy(inputs)
            inputs_2 = inputs_all[:, :2, :, :, :]
            inputs, targets = inputs_all.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            # compute output
            yhats, yhats_s = model(inputs)
            preds = F.softmax(yhats.cpu())
            softmaxs = np.max(preds.data.numpy(), axis=-1)
            n = 0
            for v in range(1,n_views+1):
                inputs = inputs_all[:, v - 1, :, :, :]
                inputs_1 = inputs_all[:, v-1, :, :, :].unsqueeze(0)
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                inputs, targets = Variable(inputs), Variable(targets)
                # compute output
                yhats, yhats_s = model(inputs)
                preds = F.softmax(yhats.cpu())
                softmaxs_1 = np.max(preds.data.numpy(), axis=-1)
                if(softmaxs_1 >= 0.9):
                    n += 1
                    if(n==1):
                        inputs_2 = inputs_1
                    else:
                        inputs_2 = torch.cat((inputs_2, inputs_1), dim=1)
                    inputs = inputs_2.cuda(device)
                    inputs = Variable(inputs)
                    # compute output
                    yhats, yhats_s = model(inputs)
                    preds = F.softmax(yhats.cpu())
                    softmaxs = np.max(preds.data.numpy(), axis=-1)
                if(n == 2):
                    break


            _, predicted = torch.max(yhats.data, 1)
            correct += (predicted.cpu() == targets.cpu()).sum()
            view_num += v
            views.append(float(v))
            total += 1
            softmaxs = softmaxs.tolist()
            softmax_list.extend(softmaxs)
            pred_list.extend(predicted.cpu().numpy())
            real_list.extend(targets.cpu().numpy())

    acc_coverage = 100 * correct / total
    mean_view = (view_num / total) - 1
    return mean_view, acc_coverage, views, softmax_list, pred_list, real_list


def mcdropout_estimation_mode(data_loader):
    # Eval using MC dropout
    model.train()
    predictions = []
    dropout_iterations = 100
    total = 0.0
    correct = 0.0
    view_num = 0.0
    views = []
    softmax_list = []
    pred_list = []
    real_list = []
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs_all = torch.from_numpy(inputs)
            inputs_2 = inputs_all[:, 0, :, :, :]
            inputs, targets = inputs_2.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            m_outputs, outputs_pool, best_view = model(inputs)
            idx = np.argsort(best_view.cpu().numpy()).flatten()
            view_number = np.argmax(best_view.cpu().numpy())
            inputs_best = inputs_all[:, view_number, :, :, :].unsqueeze(0)
            # compute output
            for v in range(2, n_views + 1):
                inputs_topk = inputs_all[:, idx[:v - 1], :, :, :]
                inputs = torch.cat((inputs_topk, inputs_best), dim=1)
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                inputs, targets = Variable(inputs), Variable(targets)
                # compute output
                pred_list_drop = []
                for d in range(dropout_iterations):
                    # compute output
                    yhats, yhats_s, _ = model(inputs)
                    preds = F.softmax(yhats.cpu())
                    preds = np.argmax(preds.data.numpy(), axis=-1)
                    pred_list_drop.extend(preds)
                pred, Mode_1 = mode(pred_list_drop)
                Mode_1 = Mode_1 / dropout_iterations
                if (Mode_1 >= thre):
                    break


            correct += (pred == targets.cpu().numpy()).sum()
            view_num += v
            views.append(float(v))
            total += 1
            Mode_1 = Mode_1.tolist()
            softmax_list.append(Mode_1)
            pred_list.extend(pred)
            real_list.extend(targets.cpu().numpy())

    acc_coverage = 100 * correct / total
    mean_view = (view_num / total) - 1

    return mean_view, acc_coverage, views, softmax_list, pred_list, real_list

def mcdropout_estimation_mean(data_loader):
    # Eval using MC dropout
    model.train()
    predictions = []
    dropout_iterations = 100
    total = 0.0
    correct = 0.0
    view_num = 0.0
    views = []
    softmax_list = []
    pred_list = []
    real_list = []
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs_all = torch.from_numpy(inputs)
            inputs_2 = inputs_all[:, 0, :, :, :]
            inputs, targets = inputs_2.cuda(device), targets.cuda(device)
            inputs, targets = Variable(inputs), Variable(targets)
            m_outputs, outputs_pool, best_view = model(inputs)
            idx = np.argsort(best_view.cpu().numpy()).flatten()
            view_number = np.argmax(best_view.cpu().numpy())
            inputs_best = inputs_all[:, view_number, :, :, :].unsqueeze(0)
            for v in range(2, n_views + 1):
                inputs_topk = inputs_all[:, idx[:v - 1], :, :, :]
                inputs = torch.cat((inputs_topk, inputs_best), dim=1)
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                inputs, targets = Variable(inputs), Variable(targets)
                # compute output
                pred_list_drop = []
                pred_list_value = []
                output_list = []
                for d in range(dropout_iterations):
                    # compute output
                    yhats, yhats_s, _ = model(inputs)
                    preds = F.softmax(yhats.cpu())
                    output_list.append(torch.unsqueeze(preds, dim=0))
                    preds_l = np.argmax(preds.data.numpy(), axis=-1)
                    pred_list_drop.extend(preds_l)
                    preds_value = np.max(preds.data.numpy(), axis=-1) # max probability value
                    pred_list_value.extend(preds_value)
                output_mean = torch.cat(output_list, 0).mean(dim=0)
                #output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
                confidence = output_mean.data.cpu().numpy().max()
                predict = output_mean.data.cpu().numpy().argmax()
                if(confidence >= thre):
                    break

            correct += (predict == targets.cpu().numpy()).sum()
            view_num += v
            views.append(float(v))
            total += 1
            confidence = confidence.tolist()
            softmax_list.append(confidence)
            pred_list.append(predict.tolist())
            real_list.extend(targets.cpu().numpy())

    acc_coverage = 100 * correct / total
    mean_view = (view_num / total) - 1

    return mean_view, acc_coverage, views, softmax_list, pred_list, real_list

def deep_ensembles_estimation(data_loader):
    # Eval using MC dropout
    model.train()
    predictions = []
    dropout_iterations = 100
    total = 0.0
    correct = 0.0
    view_num = 0.0
    views = []
    softmax_list = []
    pred_list = []
    real_list = []
    for i, (inputs, targets) in enumerate(data_loader):
        with torch.no_grad():
            # Convert from list of 3D to 4D
            inputs = np.stack(inputs, axis=1)
            inputs_all = torch.from_numpy(inputs)
            for v in range(1,n_views+1):
                inputs = inputs_all[:, :v, :, :, :]
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                inputs, targets = Variable(inputs), Variable(targets)
                # compute output
                pred_list_drop = []
                for d in range(dropout_iterations):
                    # compute output
                    yhats, yhats_s = model(inputs)
                    preds = F.softmax(yhats.cpu())
                    preds = np.argmax(preds.data.numpy(), axis=-1)
                    pred_list_drop.extend(preds)
                pred, Mode = mode(pred_list_drop)
                Mode = Mode / dropout_iterations
                if(Mode >= thre):
                    break

            correct += (pred == targets.cpu().numpy()).sum()
            view_num += v
            views.append(float(v))
            total += 1
            Mode = Mode.tolist()
            softmax_list.append(Mode)
            pred_list.extend(pred)
            real_list.extend(targets.cpu().numpy())

    acc_coverage = 100 * correct / total
    mean_view = view_num / total

    return mean_view, acc_coverage, views, softmax_list, pred_list, real_list

results_acc_mv = []
results_acc_sv = []
results_loss_mv = []
results_loss_sv = []
result_confidence = []
precision_all = []
recall_all = []
f1_all = []
accuracy_all = []
# Training / Eval loop
if args.resume:
    load_checkpoint()

def get_estimation_method(name):
    if name == "softmax":
        return softmax_estimation
    elif name == "mcdropout_mean":
        return mcdropout_estimation_mean
    elif name == "mcdropout_mode":
        return mcdropout_estimation_mode
    elif name == "deep_ensembles":
        return deep_ensembles_estimation
    else:
        print("METHOD NOT IMPLEMENTED")


uncertainty_method = get_estimation_method(args.estimation)

model.eval()
'''
avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_view_acc, avg_view_acc3, avg_view_loss = eval(test_loader)
print('\nTesting:')
print(
    '\tVal Acc of Multi-view: %.2f - Val Acc of Single view: %.2f - Loss: %.4f - M_loss: %.4f - View_acc: %.2f - View_acc3: %.2f - View_loss: %.2f' % (
        avg_test_acc.item(), avg_test_acc_single.item(),
        avg_loss.item(), avg_m_loss.item(), avg_view_acc.item(), avg_view_acc3.item(), avg_view_loss.item()))

avg_test_acc, avg_loss = eval4bestview(test_reject_loader)
print('\nTesting:')
print(
    '\tVal Acc of Multi-view: %.2f - Loss: %.4f' % (
        avg_test_acc.item(), avg_loss.item()))


'''
mean_view, acc_coverage, views, softmax_list, pred_list, real_list= uncertainty_method(test_reject_loader)
print('\nThreshold setting test:')
print('\tAcc of threshold: %.2f - Mean of view: %.2f' % (acc_coverage.item(), mean_view))


with open(args.estimation+'_views_'+str(args.threshold), "w") as file:
    file.write(str(views))
with open(args.estimation+'_list_'+str(args.threshold), "w") as file:
    file.write(str(softmax_list))
with open(args.estimation+'_pred_list_'+str(args.threshold), "w") as file:
    file.write(str(pred_list))
with open(args.estimation+'_real_list_'+str(args.threshold), "w") as file:
    file.write(str(real_list))


