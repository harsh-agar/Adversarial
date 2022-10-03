import argparse
import math
from xmlrpc.client import boolean
from tqdm import tqdm
import warnings
import os
from pathlib import Path
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from dataloader import get_cifar10, get_cifar100
from utils      import accuracy, epoch_log

from model.wrn  import WideResNet
from vat import VATLoss

import torch
import torch.optim as optim
from torch.utils.data   import DataLoader
import torch.nn as nn
from torchvision import utils as tv_utils
from torch.utils.tensorboard import SummaryWriter


def main(args):
    
    if args.dataset == "cifar10":
        args.num_classes = 10
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar10(args, 
                                                                args.datapath)
    if args.dataset == "cifar100":
        args.num_classes = 100
        labeled_dataset, unlabeled_dataset, test_dataset = get_cifar100(args, 
                                                                args.datapath)
    args.epoch = math.ceil(args.total_iter / args.iter_per_epoch)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_writer = SummaryWriter()

    if not os.path.exists(Path(args.datapath) / Path('dataloaders_%s' %(args.dataset))):
        os.makedirs(Path(args.datapath) / Path('dataloaders_%s' %(args.dataset)))

    labeled_loader_path = Path(args.datapath) / Path('dataloaders_%s' %(args.dataset)) / Path('labeled_loader.pkl')
    valid_loader_path = Path(args.datapath) / Path('dataloaders_%s' %(args.dataset)) / Path('valid_loader.pkl')
    unlabeled_loader_path = Path(args.datapath) / Path('dataloaders_%s' %(args.dataset)) / Path('unlabeled_loader.pkl')
    test_loader_path = Path(args.datapath) / Path('dataloaders_%s' %(args.dataset)) / Path('test_loader.pkl')

    if os.path.exists(labeled_loader_path) and os.path.exists(valid_loader_path) and os.path.exists(unlabeled_loader_path) and os.path.exists(test_loader_path):
        labeled_dataset = torch.load(labeled_loader_path)
        unlabeled_dataset_split = torch.load(unlabeled_loader_path)
        valid_dataset = torch.load(valid_loader_path) 
        test_dataset = torch.load(test_loader_path)
      
    else:
        val_set_idx = np.empty(0)
        for i in set(unlabeled_dataset.targets):
            data_size = int(np.where(unlabeled_dataset.targets ==  0)[0].shape[0]*0.1)
            val_set_idx = np.append(val_set_idx, np.random.choice(np.where(unlabeled_dataset.targets ==  i)[0], data_size))

        x_val = torch.empty(0, 3, 32, 32)
        y_val = torch.empty(0).int()
        x_unl = torch.empty(0, 3, 32, 32)
        y_unl = torch.empty(0).int()
        for i in tqdm(range(len(unlabeled_dataset))):
            if i in val_set_idx:
                x_val = torch.cat((x_val, unlabeled_dataset[i][0][None, :]))
                y_val = torch.cat((y_val, torch.tensor(unlabeled_dataset.targets[i : i+1])))            
            else:
                x_unl = torch.cat((x_unl, unlabeled_dataset[i][0][None, :]))
                y_unl = torch.cat((y_unl, torch.tensor(unlabeled_dataset.targets[i : i+1])))
        unlabeled_dataset_split = list(zip(x_val, y_val))
        valid_dataset = list(zip(x_unl, y_unl))
    
        torch.save(labeled_dataset, labeled_loader_path)
        torch.save(unlabeled_dataset_split, unlabeled_loader_path)
        torch.save(valid_dataset, valid_loader_path)
        torch.save(test_dataset, test_loader_path)
        

    labeled_loader      = iter(DataLoader(labeled_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    # unlabeled_loader    = DataLoader(unlabeled_dataset, 
    #                                 batch_size=args.train_batch,
    #                                 shuffle = True, 
    #                                 num_workers=args.num_workers)
    valid_loader        = DataLoader(valid_dataset, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers)
    unlabeled_loader    = iter(DataLoader(unlabeled_dataset_split, 
                                    batch_size = args.train_batch, 
                                    shuffle = True, 
                                    num_workers=args.num_workers))
    test_loader         = DataLoader(test_dataset,
                                    batch_size = args.test_batch,
                                    shuffle = False, 
                                    num_workers=args.num_workers)

    model       = WideResNet(args.model_depth, 
                                args.num_classes, widen_factor=args.model_width, dropRate=args.model_droprate)
    model       = model.to(device)

    model_wt_path = Path('model_weights_%s_%s_%s' %(args.dataset, args.num_labeled, args.model_droprate))
    logfilename     = Path(model_wt_path) / Path("log_info.txt")
    model_last_path = Path(model_wt_path) / Path("last_trained.h5")
    model_txt_path  = Path(model_wt_path) / Path("epoch_info.txt")

    start_model = 0
    if os.path.exists(model_txt_path):
      with open(model_txt_path, "r") as f:
          txt = f.read()
      start_model = int(re.search('Last model epoch: (.*)\n', txt).group(1)) + 1
      best_model = int(re.search('Best model epoch: (.*)\n', txt).group(1))
      model.load_state_dict(torch.load(model_last_path))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    
    for epoch in range(start_model, args.epoch):
        last_loss = 999999999.9
        for i in tqdm(range(args.iter_per_epoch)):
            if i % args.log_interval == 0:
                ce_losses = epoch_log()
                vat_losses = epoch_log()
                accuracies = epoch_log()

            try:
                x_l, y_l    = next(labeled_loader)
            except StopIteration:
                labeled_loader      = iter(DataLoader(labeled_dataset, 
                                            batch_size = args.train_batch, 
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_l, y_l    = next(labeled_loader)
            
            try:
                x_ul, _     = next(unlabeled_loader)
            except StopIteration:
                unlabeled_loader    = iter(DataLoader(unlabeled_dataset, 
                                            batch_size=args.train_batch,
                                            shuffle = True, 
                                            num_workers=args.num_workers))
                x_ul, _     = next(unlabeled_loader)
            
            x_l, y_l    = x_l.to(device), y_l.to(device)
            x_ul        = x_ul.to(device)

            optimizer.zero_grad()
            vatLoss = VATLoss(args)

            vat_loss = vatLoss.forward(model, x_ul)
            preds = model(x_l)
            classification_loss = criterion(preds.softmax(dim=1), y_l)
            loss = classification_loss + args.alpha * vat_loss
            loss.backward()
            optimizer.step()

            acc = accuracy(preds, y_l)

            ce_losses.update(classification_loss.item(), x_l.shape[0])
            vat_losses.update(loss.item(), x_ul.shape[0])
            accuracies.update(acc[0].item(), x_l.shape[0])

            # if i % args.log_interval == 0:
            #     print('[epoch = %d] [iteration = %d] [ce_loss: %.3f] [vat_loss: %.3f] [train_accuracy: %.3f]' %
            #     (epoch,i, ce_losses.avg, vat_losses.avg, accuracies.avg))

        val_loss = 0.0
        val_acc = epoch_log()
        for val_i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.type(torch.LongTensor).to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            acc = accuracy(outputs, labels)
            val_acc.update(acc[0].item(), inputs.shape[0])
            v_loss = criterion(outputs.softmax(dim=1), labels)
            val_loss += v_loss.item()

        if not os.path.exists(model_wt_path):
            os.makedirs(model_wt_path)

        with open(logfilename, 'a+') as f:
            f.write('[epoch = %d] train_accuracy: %.3f ce_loss: %.3f vat_loss: %.3f  val_accuracy: %.3f val_loss: %.3f \n' %
                (epoch, accuracies.avg, ce_losses.avg, vat_losses.avg, val_acc.avg, val_loss / val_i))
        print('[epoch = %d] train_accuracy: %.3f ce_loss: %.3f vat_loss: %.3f  val_accuracy: %.3f val_loss: %.3f' %
                (epoch, accuracies.avg, ce_losses.avg, vat_losses.avg, val_acc.avg, val_loss / val_i))

        train_writer.add_scalar("VAT Loss", vat_losses.avg, epoch)
        train_writer.add_scalar("Crossentropy Loss", ce_losses.avg, epoch)
        train_writer.add_scalar("Validation Loss", val_loss / val_i, epoch)
        
        model_wts_path  = Path(model_wt_path) / Path(f"epoch_{epoch}_of_{args.epoch}.h5")
        torch.save(model.state_dict(), model_last_path)

        if last_loss > val_loss:
            torch.save(model.state_dict(), model_wts_path)
            best_model = epoch
            last_loss = val_loss

        with open(model_txt_path, "w+") as f:
            f.write("Best model epoch: %d\n" % (best_model))
            f.write("Last model epoch: %d\n" % (epoch))
    train_writer.close()
    torch.save(model.state_dict(), 'model_last_path')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtual adversarial training \
                                        of CIFAR10/100 using with pytorch")
    parser.add_argument("--dataset", default="cifar10", 
                        type=str, choices=["cifar10", "cifar100"])
    parser.add_argument("--datapath", default="./data/", 
                        type=str, help="Path to the CIFAR-10/100 dataset")
    parser.add_argument('--num-labeled', type=int, 
                        default=4000, help='Total number of labeled samples')
    parser.add_argument("--lr", default=0.1, type=float,
                        help="The initial learning rate") 
    parser.add_argument("--momentum", default=0.9, type=float,
                        help="Optimizer momentum")
    parser.add_argument("--wd", default=0.0005, type=float,
                        help="Weight decay")
    parser.add_argument("--expand-labels", action="store_true", 
                        help="expand labels to fit eval steps")
    parser.add_argument('--train-batch', default=256, type=int,
                        help='train batchsize')
    parser.add_argument('--test-batch', default=256, type=int,
                        help='train batchsize')
    parser.add_argument('--total-iter', default=128*128, type=int,
                        help='total number of iterations to run')
    parser.add_argument('--iter-per-epoch', default=128, type=int,
                        help="Number of iterations to run per epoch")
    parser.add_argument('--num-workers', default=1, type=int,
                        help="Number of workers to launch during training")                        
    parser.add_argument('--alpha', type=float, default=1.0, metavar='ALPHA',
                        help='regularization coefficient (default: 0.01)')
    parser.add_argument("--dataout", type=str, default="./path/to/output/",
                        help="Path to save log files")
    parser.add_argument("--model-depth", type=int, default=28,
                        help="model depth for wide resnet") 
    parser.add_argument("--model-droprate", type=float, default=0.0,
                        help="model dropout rate for wide resnet")
    parser.add_argument("--model-width", type=int, default=2,
                        help="model width for wide resnet")
    parser.add_argument("--vat-xi", default=10.0, type=float, 
                        help="VAT xi parameter")
    parser.add_argument("--vat-eps", default=1.0, type=float, 
                        help="VAT epsilon parameter") 
    parser.add_argument("--vat-iter", default=1, type=int, 
                        help="VAT iteration parameter")
    parser.add_argument('--log-interval', type=int, default=100,
                        help='interval for logging training status')

    # parser.add_argument("--dataloader_path", default="./data/dataloaders/", 
    #                 type=str, help="Path to the saved model")
    # Add more arguments if you need them
    # Describe them in help
    # You can (and should) change the default values of the arguments
    
    args = parser.parse_args()


    main(args)