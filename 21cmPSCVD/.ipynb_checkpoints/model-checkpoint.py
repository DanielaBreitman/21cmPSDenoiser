import torch
from torch import nn

def modulelist2sequential(module:nn.ModuleList) -> nn.Sequential:
    out = nn.Sequential()
    for layer in module:
        out.append(layer)
    return out

class ResNet(torch.nn.Module):
    def __init__(self, module, act_fn: object = nn.LeakyReLU):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(0.1)
        self.act = act_fn()

    def forward(self, inputs):
        return (self.module(inputs) + inputs).float()

class CNN(nn.Module):
    def __init__(self, nconvs:int, 
                 in_ch: int, out_ch: int, dropout:bool=False,
                 hid_ch:int=None, kernel_size:int = 2, f_dropout:float=0.1,
                 stride:int=1, padding:int='same', final_act:bool=False,
                 batch_norm: bool = False, 
                 act_fn: object = nn.LeakyReLU, residual:bool=False):
        super().__init__()
        self.cnn = cnn_list(nconvs=nconvs, 
                 in_ch=in_ch, out_ch=out_ch, 
                 hid_ch=hid_ch, kernel_size=kernel_size, 
                 stride=stride, padding=padding, final_act=final_act,
                 batch_norm=batch_norm, act_fn=act_fn, residual=residual)
        self.residual = residual
        
    def forward(self, x):
        y = self.cnn(x)
        return y

def cnn_list(nconvs:int, 
                 in_ch: int, out_ch: int, dropout:bool=False,
                 hid_ch:int=None, kernel_size:int = 2, f_dropout:float=0.1,
                 stride:int=1, padding:int='same', final_act:bool=False,
                 batch_norm: bool = False, act_fn: object = nn.LeakyReLU, 
                 residual:bool = False) -> nn.ModuleList:

    act = act_fn()
    if hid_ch is None:
        hid_ch = out_ch
    conv_in = nn.Sequential(nn.Conv2d(in_ch, hid_ch, kernel_size = kernel_size, stride = stride, padding = padding), act_fn())
    conv_hid = nn.Sequential(nn.Conv2d(hid_ch, hid_ch, kernel_size = kernel_size, stride = stride, padding = padding), act_fn())
    if batch_norm:
        conv_in.append(nn.BatchNorm2d(hid_ch))
        conv_hid.append(nn.BatchNorm2d(hid_ch))
    if dropout:
        conv_in.append(nn.Dropout(f_dropout))
        conv_hid.append(nn.Dropout(f_dropout))
    cnn = modulelist2sequential(nn.ModuleList([conv_hid for i in range(nconvs-2)]))
    if final_act:
        conv_out = nn.Sequential(nn.Conv2d(hid_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding), act_fn())
    else:
        conv_out = nn.Conv2d(hid_ch, out_ch, kernel_size = kernel_size, stride = stride, padding = padding)
    if residual:
        return ResNet(nn.Sequential(conv_in.extend(cnn.append(conv_out))))
    else:
        return conv_in.extend(cnn.append(conv_out))
        

class Autoencoder(nn.Module):
    def __init__(self, in_ch: int=2, out_ch:int=2):
        """
        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.cnn_in = CNN(nconvs = 2, in_ch = in_ch, 
                              out_ch = 128, kernel_size = 5, padding = 'same', 
                              batch_norm = True, final_act = True, residual = False)
        self.resblock32 = CNN(nconvs = 2, in_ch = 128, 
                              out_ch = 128, kernel_size = 5, padding = 'same', 
                              batch_norm = True, final_act = True, residual = True)
        self.resblock32_2 = CNN(nconvs = 2, in_ch = 128, 
                              out_ch = 128, kernel_size = 3, padding = 'same', 
                              batch_norm = True, final_act = True, residual = True)
        self.resblock64 = CNN(nconvs = 2, in_ch = 128, 
                              out_ch = 128, kernel_size = 2, padding = 'same', 
                              batch_norm = True, final_act = True, residual = True)
        self.resblock128 = CNN(nconvs = 2, in_ch = 128, 
                              out_ch = 128, kernel_size = 2, padding = 'same', 
                              batch_norm = True, final_act = True, residual = True)
        self.cnn1 = CNN(nconvs = 2, in_ch = 128, 
                              out_ch = 128, kernel_size = 3, padding = 'same', 
                              batch_norm = False, final_act = True, residual = True)
        self.cnn2 = CNN(nconvs = 2, in_ch = 128, 
                              out_ch = 128, kernel_size = 3, padding = 'same', 
                              batch_norm = False, final_act = True, residual = True)
        self.cnn3 = CNN(nconvs = 4, in_ch = 128, 
                              out_ch = 128, kernel_size = 2, padding = 'same', 
                              batch_norm = False, final_act = True, residual = True)
        self.cnn4 = CNN(nconvs = 4, in_ch = 128, 
                              out_ch = 128, kernel_size = 2, padding = 'same', 
                              batch_norm = False, final_act = True, residual = True)
        self.maxpool = nn.MaxPool2d(2)
        

        ## DECODER ##
        self.resnet1 = CNN(nconvs = 2, in_ch = 128, 
                           out_ch = 128, kernel_size = 2, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)
    
        self.cnn7 = CNN(nconvs = 4, in_ch = 128, hid_ch = 128,
                           out_ch = 128, kernel_size = 2, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)

        self.resnet2 = CNN(nconvs = 2, in_ch = 128,
                           out_ch = 128, kernel_size = 2, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)
        self.cnn8 = CNN(nconvs = 4, in_ch = 128, hid_ch = 128,
                           out_ch = 128, kernel_size = 2, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)
        self.resnet3 = CNN(nconvs = 2, in_ch = 128,
                           out_ch = 128, kernel_size = 2, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)

        self.cnn9 = CNN(nconvs = 2, in_ch = 128, hid_ch = 128,
                           out_ch = 128, kernel_size = 6, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)
        self.resnet4 = CNN(nconvs = 2, in_ch = 128,
                           out_ch = 128, kernel_size = 6, padding = 'same', 
                           batch_norm = False, final_act = True, residual = True)

        self.cnn10 = CNN(nconvs = 2, in_ch = 128, hid_ch = 16,
                           out_ch = out_ch, kernel_size = 6, padding = 'same', 
                           batch_norm = False, final_act = False, residual = False)
        self.upsample = nn.Upsample(scale_factor = 2)
    def forward(self, x):
        ## ENCODER ##
        encode1 = self.cnn_in(x)
        encode2 = self.resblock32(encode1)
        encode3 = self.maxpool(encode2)
        encode4 = self.resblock32_2(encode3)
        encode5 = self.cnn1(encode4)
        encode6 = self.cnn2(encode5)
        encode7 = self.maxpool(encode6)
        encode8 = self.resblock64(encode7)
        encode9 = self.cnn3(encode8)
        encode10 = self.maxpool(encode9) 
        
        ## Bottleneck
        encode11 = self.resblock128(encode10) 
        encode = self.cnn4(encode11)
        
        decode1 = self.resnet1(encode) + encode11
        
        ## DECODER ##
        decode2 = self.upsample(decode1)
        decode3 = self.cnn7(decode2) + encode9
        decode4 = self.resnet2(decode3) + encode8 
        decode5 = self.upsample(decode4) + encode6
        decode6 = self.cnn8(decode5) + encode5
        decode7 = self.resnet3(decode6) + encode4
        decode8 = self.upsample(decode7) + encode1
        decode9 = self.cnn9(decode8) + encode1
        decode10 = self.resnet4(decode9) + encode1
        decoded = self.cnn10(decode10)
        return decoded