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

model2 = resnet18(pretrained=args.pretrained, num_classes=len(classes))
model2.to(device)
model3 = resnet34(pretrained=args.pretrained, num_classes=len(classes))
model3.to(device)
model4 = resnet50(pretrained=args.pretrained, num_classes=len(classes))
model4.to(device)
model5 = resnet101(pretrained=args.pretrained, num_classes=len(classes))
model5.to(device)
model.to(device)
#print(summary(model, (12, 3, 224, 224)))
cudnn.benchmark = True

#logger = Logger('logs')
# Loss and Optimizer
lr = args.lr
n_epochs = args.epochs
criterion = nn.CrossEntropyLoss()
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


def train():
    train_size = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        # Convert from list of 3D to 4D
        inputs = np.stack(inputs, axis=1)

        inputs = torch.from_numpy(inputs)
        #inputs = inputs[:,:2,:,:,:]
        inputs, targets = inputs.cuda(device), targets.cuda(device)
        inputs, targets = Variable(inputs), Variable(targets)
        #print(targets.size())
        # compute output
        m_outputs, outputs_pool = model(inputs)
        m_loss = criterion(m_outputs, targets)
        s_loss = 0.0
        for v in outputs_pool:
            s_loss += criterion(v, targets)
        loss = m_loss + s_loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % args.print_freq == 0:
            print("\tIter [%d/%d] Loss: %.4f M_loss: %.4f S_loss: %.4f" % (i + 1, train_size, loss.item(), m_loss.item(), s_loss.item()))


# Validation and Testing
def eval(data_loader, model):
    # Eval
    model.eval()
    total = 0.0
    correct = 0.0
    correct_single = 0.0

    total_loss = 0.0
    total_m_loss = 0.0
    total_s_loss = 0.0
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
            m_outputs, outputs_pool = model(inputs)
            m_loss = criterion(m_outputs, targets)
            s_loss = 0.0
            for v in outputs_pool:
                s_loss += criterion(v, targets)
            loss = m_loss + s_loss
            total_m_loss += m_loss
            total_s_loss += s_loss
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
    avg_test_acc_single = 100 * correct_single / (total * len(outputs_pool))

    return avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_s_loss
        #, precision, recall, f1, accuracy

def val_acqu(data_loader):
    # Eval using MC dropout
    model.train()
    predictions = []
    dropout_iterations = 50
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


def deep_ensembles_estimation(data_loader):
    model.eval()
    model2.eval()
    model3.eval()
    predictions = []
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
            print(inputs_all.size(1))
            for v in range(1,n_views+1):
                inputs = inputs_all[:, :v, :, :, :]
                inputs, targets = inputs.cuda(device), targets.cuda(device)
                inputs, targets = Variable(inputs), Variable(targets)
                # compute output
                pred_list_value = []
                output_list = []

                yhats, yhats_s = model(inputs)
                preds = F.softmax(yhats.cpu())
                output_list.append(torch.unsqueeze(preds, dim=0))
                yhats, yhats_s = model2(inputs)
                preds = F.softmax(yhats.cpu())
                output_list.append(torch.unsqueeze(preds, dim=0))
                yhats, yhats_s = model3(inputs)
                preds = F.softmax(yhats.cpu())
                output_list.append(torch.unsqueeze(preds, dim=0))
                yhats, yhats_s = model4(inputs)
                preds = F.softmax(yhats.cpu())
                output_list.append(torch.unsqueeze(preds, dim=0))
                yhats, yhats_s = model5(inputs)
                preds = F.softmax(yhats.cpu())
                output_list.append(torch.unsqueeze(preds, dim=0))

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
            pred_list.append(predict)
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
    # Load checkpoint 2
    print('\n==> Loading model 2 checkpoint..')
    checkpoint2 = torch.load('/home/ga62rup/models/Uncertainty_mvcnn/checkpoint/resnet18_checkpoint.pth.tar')
    best_acc2 = checkpoint2['best_acc']
    start_epoch2 = checkpoint2['epoch']
    model2.load_state_dict(checkpoint2['state_dict'])
    # Load checkpoint 3
    print('\n==> Loading model 3 checkpoint..')
    checkpoint3 = torch.load('/home/ga62rup/models/Uncertainty_mvcnn/checkpoint/resnet34_checkpoint.pth.tar')
    best_acc3 = checkpoint3['best_acc']
    start_epoch3 = checkpoint3['epoch']
    model3.load_state_dict(checkpoint3['state_dict'])
    print('\n==> Loading model 4 checkpoint..')
    checkpoint4 = torch.load('/home/ga62rup/models/Uncertainty_mvcnn/checkpoint/resnet50_checkpoint.pth.tar')
    best_acc4 = checkpoint4['best_acc']
    start_epoch4 = checkpoint4['epoch']
    model4.load_state_dict(checkpoint4['state_dict'])
    print('\n==> Loading model 5 checkpoint..')
    checkpoint5 = torch.load('/home/ga62rup/models/Uncertainty_mvcnn/checkpoint/resnet101_checkpoint.pth.tar')
    best_acc5 = checkpoint5['best_acc']
    start_epoch5 = checkpoint5['epoch']
    model5.load_state_dict(checkpoint5['state_dict'])



def get_estimation_method(name):
    if name == "deep_ensembles":
	return deep_ensembles_estimation
    else:
        print("METHOD NOT IMPLEMENTED")


uncertainty_method = get_estimation_method(args.estimation)

# testing results
avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_s_loss = eval(test_loader, model)
print('\nTesting:')
print(
    '\tVal Acc of Multi-view: %.2f - Val Acc of Single view: %.2f - Loss: %.4f - M_loss: %.4f - S_loss: %.4f' % (
        avg_test_acc.item(), avg_test_acc_single.item(),
        avg_loss.item(), avg_m_loss.item(), avg_s_loss.item()))
avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_s_loss = eval(test_loader, model2)
print('\nTesting:')
print(
    '\tVal Acc of Multi-view: %.2f - Val Acc of Single view: %.2f - Loss: %.4f - M_loss: %.4f - S_loss: %.4f' % (
        avg_test_acc.item(), avg_test_acc_single.item(),
        avg_loss.item(), avg_m_loss.item(), avg_s_loss.item()))
avg_test_acc, avg_test_acc_single, avg_loss, avg_m_loss, avg_s_loss = eval(test_loader, model3)
print('\nTesting:')
print(
    '\tVal Acc of Multi-view: %.2f - Val Acc of Single view: %.2f - Loss: %.4f - M_loss: %.4f - S_loss: %.4f' % (
        avg_test_acc.item(), avg_test_acc_single.item(),
        avg_loss.item(), avg_m_loss.item(), avg_s_loss.item()))

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

