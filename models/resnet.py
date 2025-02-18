import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_noshortcut(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck_noshortcut(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_noshortcut, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_cifar, self).__init__()
        self.in_planes = 16

        self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class WResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, k, num_classes=10):
        super(WResNet_cifar, self).__init__()
        self.in_planes = 16*k

        self.conv1 = nn.Conv2d(3, 16*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16*k)
        self.layer1 = self._make_layer(block, 16*k, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32*k, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64*k, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*k*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ImageNet models
def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet18_noshort():
    return ResNet(BasicBlock_noshortcut, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet34_noshort():
    return ResNet(BasicBlock_noshortcut, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet50_noshort():
    return ResNet(Bottleneck_noshortcut, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet101_noshort():
    return ResNet(Bottleneck_noshortcut, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNet152_noshort():
    return ResNet(Bottleneck_noshortcut, [3,8,36,3])

# CIFAR-10 models
def ResNet20(num_classes=10):
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes=num_classes)

def ResNet20_noshort():
    depth = 20
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet32(num_classes=10):
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes=num_classes)

def ResNet32_noshort():
    depth = 32
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet44(num_classes=10):
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes=num_classes)

def ResNet44_noshort():
    depth = 44
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet50_16_noshort():
    depth = 50
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet56(num_classes=10):
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes=num_classes)

def ResNet56_noshort():
    depth = 56
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def ResNet110(num_classes=10):
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock, [n,n,n], num_classes=num_classes)

def ResNet110_noshort():
    depth = 110
    n = (depth - 2) // 6
    return ResNet_cifar(BasicBlock_noshortcut, [n,n,n])

def WRN56_2(num_classes=10):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 2, num_classes=num_classes)

def WRN56_4(num_classes=10):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 4, num_classes=num_classes)

def WRN56_8(num_classes=10):
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock, [n,n,n], 8, num_classes=num_classes)

def WRN56_2_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2)

def WRN56_4_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4)

def WRN56_8_noshort():
    depth = 56
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 8)

def WRN110_2_noshort():
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 2)

def WRN110_4_noshort():
    depth = 110
    n = (depth - 2) // 6
    return WResNet_cifar(BasicBlock_noshortcut, [n,n,n], 4)

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))


class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class share(nn.Module):
    def __init__(self, input, hidden1, hidden2):
        super(share, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input, hidden1), nn.ReLU(inplace=True))

    def forward(self, x):
        output = self.layer(x)

        return output


class task(nn.Module):
    def __init__(self, hidden2, output, num_classes):
        super(task, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_classes):
            self.layers.append(nn.Sequential(nn.Linear(hidden2, output), nn.Sigmoid()))

    def forward(self, x, num, c):
        si = x.shape[0]
        output = torch.tensor([]).cuda()
        for i in range(si):
            output = torch.cat((output, self.layers[c[num[i]]](x[i].unsqueeze(0))), 0)

        return output

class VNet(nn.Module):
    def __init__(self, input, hidden1, output):
        super(VNet, self).__init__()
        self.linear1 = nn.Linear(input, hidden1)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)

        return F.sigmoid(out)


class CVNet(nn.Module):
    def __init__(self, input, hidden1, output, num_classes):
        super(CVNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_classes):
            self.layers.append(nn.Sequential( nn.Linear(input, hidden1), nn.ReLU(inplace=True), nn.Linear(hidden1, output), nn.Sigmoid() ))

    def forward(self, x, num, c):
        si = x.shape[0]
        output = torch.tensor([]).cuda()
        for i in range(si):
            # output = torch.cat(( output, self.layers[num[i]]( (x[i].unsqueeze(0)) ) ),0)
            output = torch.cat((output, self.layers[c[num[i]]](x[i].unsqueeze(0))), 0)

        return output


class ACVNet(nn.Module):
    def __init__(self, input, hidden1, hidden2, output, num_classes):
        super(ACVNet, self).__init__()
        self.feature = share(input, hidden1, hidden2)
        self.classfier = task(hidden2, output, num_classes)

    def forward(self, x, num, c):
        # num = torch.argmax(num, -1)
        output = self.classfier( self.feature(x), num, c )

        return output
