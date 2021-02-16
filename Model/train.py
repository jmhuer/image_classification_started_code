import torchvision
from .models import CNNClassifier, LastLayer_Alexnet, save_model, load_weights
import torch
# import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data, accuracy
from .logging import launchTensorBoard, train_log_action, val_log_action
from .transforms import image_transforms, tensor_transform


def train(args):
    from os import path
    '''
     logging
     '''
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        launchTensorBoard(dir=args.log_dir)
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'), flush_secs=1)
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'), flush_secs=1)
    '''
    GPU OR CPU?
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    '''
    Below we define the model
    if continue training args we load weights
    if cnt training then load weights
    '''
    model = CNNClassifier()
    # model = LastLayer_Alexnet()
    if args.continue_training: load_weights(model, args.continue_training)
    model = model.to(device)
    '''
    optimizer
    '''
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    '''
    loss
    '''
    # loss = torch.nn.BCEWithLogitsLoss().to(device)
    # loss = torch.nn.MSELoss().to(device)
    loss = torch.nn.CrossEntropyLoss().to(device)
    '''
    load data
    return dictionary with train and valid items. For example data['train']
    assumes dataset is a folder with train and val subfolders
    '''
    data = load_data('Collect_data/dataset', transforms=tensor_transform, num_workers=1)
    '''
    train
    '''
    global_step = 0
    for epoch in range(args.num_epoch):
        model.train()
        for img, label in data['train']:

            # torchvision.utils.save_image(img[0], "test1.jpg")

            label = label.to(device)
            img = img.to(device)
            logit = model(img)


            loss_train = loss(logit, label)
            '''
            add image to logger every 10 batches
            '''
            if train_logger is not None:
                train_log_action(train_logger, img[0].detach().cpu(), logit[0].detach().cpu(), loss_train.detach(),
                                 global_step)

            print('epoch %-3d \t loss = %0.6f' % (epoch, loss_train))
            '''
            update weights below
            '''
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            global_step += 1

        ##now we validate
        model.eval()
        loss_total_val = 0
        cnt = 0
        scores = []
        for img, label in data['val']:
            label = label.to(device)
            img = img.to(device)
            logit = model(img)
            loss_val = loss(logit, label)
            loss_total_val += loss_val.detach()
            scores.append(accuracy(logit, label))
            cnt += 1.0

        if valid_logger is not None:
            total_loss = float(loss_total_val) / cnt
            val_log_action(valid_logger, img[0].detach().cpu(), logit[0].detach().cpu(), total_loss, global_step)

        print("\n")
        print('epoch %-3d \t va_loss = %0.6f  \t accuracy %0.6f' % (epoch, total_loss, sum(scores) / len(scores)))
        print("\n")
        info = "epoch" + str(epoch) + "_" + str(round(float(sum(scores) / len(scores)), 2))
        save_model(model, info)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-w', '--weight_decay', type=float, default=1e-5)
    parser.add_argument('-c', '--continue_training', type=str)

    args = parser.parse_args()

    train(args)
