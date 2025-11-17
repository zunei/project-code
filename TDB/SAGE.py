import copy
import os

import torch
from torch import optim, nn
from models import *
from utils import *
modelsave = './saved_model/'
if not os.path.exists(modelsave):
    os.makedirs(modelsave)
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)
random.seed(10)


def SAGE(model, train_loader, args,device):
    for param in list(model.parameters()):
        param.requires_grad = True
    model=model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().cuda()
    model.train()

    for epoch in range(args.SAGE_epochs):
        print("SAGE的第" + str(epoch) + '轮')
        train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            target=target.long()
            optimizer.zero_grad()
            feature1, feature2, feature3, _, output = model(data)
            loss1 = criterion(output, target)
            loss2 = at_gen(feature1.detach(), feature2)
            loss3 = at_gen(feature2.detach(), feature3)
            loss=loss1+args.drate1*loss2+args.drate2*loss3
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        # torch.save(model.state_dict(), modelsave + 'student_model.pth')

    return model

