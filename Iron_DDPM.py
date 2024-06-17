import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optimembedding_dim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from numpy.random import randn
import torchvision.utils
from torch.distributions import uniform
from mpl_toolkits.axes_grid1 import ImageGrid

import os
import copy
import utils
import torch.nn.parallel

# 导入pandas库，用于处理数据表
import pandas as pd

current_dir = os.getcwd()
# 训练数据文件路径
whole_dir = 'G:/DATA/JinShu/Clip'
# 模型保存文件
save_dir = str(current_dir) + '/model/All_DDPM_Irons_gen.pth.tar'

# 加载模型文件，可以和上面的一致
load_dir = str(current_dir) + '/model/All_DDPM_Irons.pth.tar'

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.backends.cuda.max_split_size_mb = 1000

#device_ids = [0, 1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'

is_predict = False

batch_size = 1                  #训练批图像数量，根据GPU能力修改
n_sampled_images = 4            #类别标签数量
n_epoch = 200                   #训练迭代次数
n_ax = int(n_epoch/20)          #模型保存的周期
learning_rate = 3e-4            #学习率
total_loss_min = np.Inf         #总的损失
image_shape = (1,512,512)       #输入图片的尺寸
image_size = 512                #图片的尺寸
image_dim = int(np.prod(image_shape))   #图片的维度


process = 15                    #第一阶段温度[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14] 表示温度700-860
temperature = 2                 #第二阶段温度[0, 1] 表示温度500-550
cooling_1 = 3                   #铸件第一阶段冷却时间[0,1,2] 现在只有0 表示2h
cooling_2 = 3                   #铸件第二阶段冷却时间[0,1,2] 现在只有0 表示3h
forging_temps = 3               #锻造温度（备用）
heat_treatments = 2             #预锻热处理（备用）
magnifications = 6              #图像放大倍率（备用）

embedding_dim = 100             #嵌入层的维度大小
num_classes = 28                #类别数量，如果类别有变化，需要修改此处


fig = plt.figure(figsize=(100, 100))


#第一阶段温度
tempure_1 = "710"
#第二阶段温度
tempure_2 = "500"
#第一阶段冷却时长
cooling1 = "2h"
#第二阶段冷却时长
cooling2 = "3h"
#预留参数
para1 = "p11"
#预留参数
para2 = "p21"
#预留参数
para3 = "p31"

# 第一阶段温度字典
temp_1_dict = {'710': 0, '720': 1, '730': 2, '740': 3, '750': 4, '760': 5, '770': 6,
               '780': 7, '790': 8, '800': 9, '810': 10, '820': 11, '830': 12,
               '840': 13, '860': 14}
# 第二阶段温度字典
temp_2_dict = {'500': 0, '550': 1}
# 第一阶段冷却时间字典，现在只用3h,其他预留
cooling_1_dict = {'2h': 0, '3h': 0, '4h': 0}
# 第二阶段冷却时间字典，现在只用2h,其他预留
cooling_2_dict = {'3h': 0, '2h': 0, '4h': 0}
# 预留参数1
para1_dict = {'p11': 0, 'p12': 0}
# 预留参数2
para2_dict = {'p21': 0, 'p22': 1, 'p23': 0}
# 预留参数3
para3_dict = {'p31': 0, 'p32': 0, 'p33': 0, 'p34': 0, 'p35': 0, 'p36': 0}


#数据类别
class_table = torch.tensor([[0., 0., 1., 1., 2., 2., 3., 3., 4., 4., 5., 5., 6., 6., 7., 7., 8., 8.,
         9., 9., 10., 10., 11., 11., 12., 13., 14., 14.],
        [0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
         0., 1., 0., 1., 0., 1., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

# print(class_table)
# print(class_table.shape)
# exit()

#分类标签
label_dict = {
    0: '710-500', 1: '710-550', 2: '720-500', 3: '720-550', 4: '730-500', 5: '730-550',
    6: '740-500', 7: '740-550', 8: '750-500', 9: '750-550', 10: '760-500', 11: '760-550',
    12: '770-500', 13: '770-550', 14: '780-500', 15: '780-550', 16: '790-500', 17: '790-550',
    18: '800-500', 19: '800-550', 20: '810-500', 21: '810-550', 22: '820-500', 23: '820-550',
    24: '830-500', 25: '840-500', 26: '860-500', 27: '860-550'
}

df = pd.read_excel('data/updated_data.xlsx')


grid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Lambda(lambda t: (t * 2) - 1)
    ])

whole_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.RandomCrop(512),
    transforms.Lambda(lambda t: (t * 2) - 1)
    ])

train_dataset = datasets.ImageFolder(whole_dir, transform=whole_transform)

aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    ]) 

reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t * 255.),
    ])

#数据加载
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# 定义一个函数，接受一个字典，和两个数字作为参数
def get_label_idx(label_dict, num1, num2):
    # 将两个数字拼接成一个字符串，用"-"分隔
    label = str(num1) + "-" + str(num2)
    print(label)
    # 判断字典中是否有这个label对应的idx
    if label in label_dict.values():
        # 如果有，返回idx
        return list(label_dict.values()).index(label)
    else:
        # 如果没有，返回-1
        return -1


def find_closest_match(data, target, column):
    # 计算目标列中每个值和目标值的绝对差值
    diff = abs(data[column] - target)
    # 找出差值最小的那一行的索引
    index = diff.idxmin()
    # 返回那一行的所有数据，包括目标列，作为一个列表
    return data.iloc[index].values.flatten().tolist()


# result = find_closest_match(df, 1221.15, 'UTS')
# print(result)
# print(round(result[4], -1), result[6])
# idx = get_label_idx(label_dict, round(result[4], -1), result[6])
# print(idx)
# test_labels = [[idx]]
#
# print(test_labels)
# # 调用函数，传入数据表，目标值，和目标列作为参数
# # UTS	Elongation
# result = find_closest_match(df, 1221.15, 'UTS')
# print(result)
# exit()


#创建类别及标签
def class_maker(batch_size, labels, class_table):
    Temp1 = torch.zeros([batch_size])
    Temp2 = torch.zeros([batch_size])
    Cooling1 = torch.zeros([batch_size])
    Cooling2 = torch.zeros([batch_size])
    SK = torch.zeros([batch_size])
    HT = torch.zeros([batch_size])
    Mag = torch.zeros([batch_size])
    for i in range(batch_size):
        Temp1[i] = class_table[0,labels[i]]
        Temp2[i] = class_table[1,labels[i]]
        Cooling1[i] = class_table[2,labels[i]]
        SK[i] = class_table[3,labels[i]]
        HT[i] = class_table[4,labels[i]]
        Cooling2[i] = class_table[5,labels[i]]
        Mag[i] = class_table[6,labels[i]]

    return Temp1, Temp2, Cooling1, SK, HT, Cooling2, Mag

#显示图像
def show_images(images, index, label):
    ax = fig.add_subplot(21, 1, index + 1, xticks=[], yticks=[])
    plt.gca().set_title(label)
    # ax.set_title(label,fontsize = 40)
    plt.imshow(images.cpu(), cmap='gray')
    plt.show()

def show_predict_images(images, labels, label_dict, files):
    labels_title = []
    for i in labels:
        labels_title.append(label_dict[i.item()])
    # 创建一个新的figure对象，用于保存每个图片
    fig = plt.figure()
    # 遍历images和labels_title，分别显示和保存每个图片
    for i, (im, title) in enumerate(zip(images.cpu().view(-1, image_size, image_size), labels_title)):
        # 清除当前的axes
        plt.clf()
        # 在当前的axes上显示图片，使用灰度色彩映射
        plt.imshow(im, cmap = 'gray')
        # 设置图片的标题，使用title参数
        plt.title(title)
        # 设置标题的字体大小为28
        plt.title.set_size(28)
        # 生成图片的文件名，使用n_epoch和i参数
        figname = files[i] #str(current_dir) + '/Predict-Images/'  + '_' + str(i)
        # 保存图片到文件，去除多余的空白边缘
        fig.savefig(figname, bbox_inches='tight')
    # 关闭figure对象
    plt.close(fig)


#显示生成图像
def show_grids(images, labels, n_epoch, label_dict):
    labels_title = []
    for i in labels:
        labels_title.append(label_dict[i.item()])
    fig = plt.figure(figsize=(40., 40.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 2),
                     axes_pad=0.5
                     )
    j = 0
    for ax, im in zip(grid, images.cpu().view(-1,image_size,image_size)):
        ax.imshow(im, cmap = 'gray')
        ax.title.set_text(labels_title[j])
        ax.title.set_size(28)
        #fig.gca().set_title(labelt[i])
        j += 1
    #plt.show()
    figname = str(current_dir) + '/Generated-Images/' + str(1200+n_epoch)
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)


def show_image(images):
    fig = plt.figure(figsize=(6.61, 6.61))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(1, 1),
                     axes_pad=0.5
                     )
    plt.axis('off')
    j = 0
    for ax, im in zip(grid, images.cpu().view(-1, image_size, image_size)):
        ax.imshow(im, cmap='gray')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        j += 1

    plt.show()
    figname = str('Synthesize')
    fig.savefig(figname, bbox_inches='tight')
    plt.close(fig)



#模型保存
def save_model(address):
    checkpoint = {"model_state": model.state_dict(),
                   "ema_model_state": ema_model.state_dict(),
              "model_optimizer": optimizer.state_dict()}
    torch.save(checkpoint, address)

#模型加载
def load_model(address):
    if torch.cuda.is_available() == True:
        checkpoint = torch.load(address)
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["model_optimizer"])
    else:
        checkpoint = torch.load(address, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        ema_model.load_state_dict(checkpoint["ema_model_state"])
        optimizer.load_state_dict(checkpoint["model_optimizer"])



#网络模型
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=image_size):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, SHP, Loc, CR, SK, HT, FT, Mag, cfg_scale=0):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(device)
                predicted_noise = model(x, t, SHP, Loc, CR, SK, HT, FT, Mag)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None, None, None, None, None, None, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                            x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        # x = (x.clamp(-1,1)+1)/2
        # x = (x*255).type(torch.uint8)
        return x


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, (out_channels - 7))
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.SHP_label = nn.Sequential(
            nn.Embedding(process, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.Loc_label = nn.Sequential(
            nn.Embedding(temperature, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.CR_label = nn.Sequential(
            nn.Embedding(cooling_1, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.SK_label = nn.Sequential(
            nn.Embedding(cooling_2, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.HT_label = nn.Sequential(
            nn.Embedding(heat_treatments, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.FT_label = nn.Sequential(
            nn.Embedding(forging_temps, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.Mag_label = nn.Sequential(
            nn.Embedding(magnifications, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))

    def forward(self, x, t, TEMP1, TEMP2, COOLING1, COOLING2, PARA1, PARA2, PARA3):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        temp1emb = self.SHP_label(TEMP1).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        temp2emb = self.Loc_label(TEMP2).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        cooling1emb = self.CR_label(COOLING1).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        cooling2emb = self.SK_label(COOLING2).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        para1emb = self.HT_label(PARA1).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        para2emb = self.FT_label(PARA2).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        para3emb = self.Mag_label(PARA3).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, temp1emb, temp2emb, cooling1emb, cooling2emb, para1emb, para2emb, para3emb), dim=1)
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, emb_dim=512):
        super().__init__()
        self.imsize = imsize
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels - 7, in_channels // 2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels)
        )
        self.SHP_label = nn.Sequential(
            nn.Embedding(process, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.Loc_label = nn.Sequential(
            nn.Embedding(temperature, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.CR_label = nn.Sequential(
            nn.Embedding(cooling_1, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.SK_label = nn.Sequential(
            nn.Embedding(cooling_2, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.HT_label = nn.Sequential(
            nn.Embedding(heat_treatments, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.FT_label = nn.Sequential(
            nn.Embedding(forging_temps, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))
        self.Mag_label = nn.Sequential(
            nn.Embedding(magnifications, embedding_dim),
            nn.Linear(embedding_dim, 1 * self.imsize * self.imsize))

    def forward(self, x, skip_x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        shpemb = self.SHP_label(SHP).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        locemb = self.Loc_label(Loc).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        cremb = self.CR_label(CR).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        skemb = self.SK_label(SK).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        htemb = self.HT_label(HT).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        ftemb = self.FT_label(FT).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        magemb = self.Mag_label(Mag).view(x.shape[0], 1, x.shape[-2], x.shape[-1])
        x = torch.cat((x, shpemb, locemb, cremb, skemb, htemb, ftemb, magemb), dim=1)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_conditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=512, num_classes=None):
        super().__init__()
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32, 256)
        self.sa1 = SelfAttention(32, 256)
        self.down2 = Down(32, 64, 128)
        self.sa2 = SelfAttention(64, 128)
        self.down3 = Down(64, 128, 64)
        self.sa3 = SelfAttention(128, 64)
        self.down4 = Down(128, 256, 32)
        self.sa4 = SelfAttention(256, 32)
        self.down5 = Down(256, 512, 16)
        self.sa5 = SelfAttention(512, 16)
        self.down6 = Down(512, 512, 8)
        self.sa6 = SelfAttention(512, 8)

        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 512)

        self.up6 = Up(1024, 256, 16)
        self.as6 = SelfAttention(256, 16)
        self.up5 = Up(512, 128, 32)
        self.as5 = SelfAttention(128, 32)
        self.up4 = Up(256, 64, 64)
        self.as4 = SelfAttention(64, 64)
        self.up3 = Up(128, 32, 128)
        self.as3 = SelfAttention(32, 128)
        self.up2 = Up(64, 16, 256)
        self.as2 = SelfAttention(16, 256)
        self.up1 = Up(32, 8, 512)
        self.as1 = SelfAttention(8, 512)
        self.outc = nn.Conv2d(8, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
                10000
                ** (torch.arange(0, channels, 2).float().to(device) / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, SHP, Loc, CR, SK, HT, FT, Mag):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x0 = self.inc(x)
        x1 = self.down1(x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        # x1 = self.sa1(x1)
        x2 = self.down2(x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        # x2 = self.sa2(x2)
        x3 = self.down3(x2, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x3 = self.sa3(x3)
        x4 = self.down4(x3, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x4 = self.sa4(x4)
        x5 = self.down5(x4, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x5 = self.sa5(x5)
        x6 = self.down6(x5, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x6 = self.sa6(x6)

        x6 = self.bot1(x6)
        x6 = self.bot2(x6)
        x6 = self.bot3(x6)

        x = self.up6(x6, x5, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as6(x)
        x = self.up5(x, x4, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as5(x)
        x = self.up4(x, x3, t, SHP, Loc, CR, SK, HT, FT, Mag)
        x = self.as4(x)
        x = self.up3(x, x2, t, SHP, Loc, CR, SK, HT, FT, Mag)
        # x = self.as3(x)
        x = self.up2(x, x1, t, SHP, Loc, CR, SK, HT, FT, Mag)
        # x = self.as2(x)
        x = self.up1(x, x0, t, SHP, Loc, CR, SK, HT, FT, Mag)
        # x = self.as1(x)
        output = self.outc(x)

        return output

def get_string_timestamp():
    # 使用time.time()函数，获取当前的时间戳，是一个浮点数
    ts = time.time()
    # 使用str()函数，将浮点数转换为字符串
    return str(ts)


# 生成图片
def predict():
    with torch.no_grad():
        test_labels = torch.randint(0, num_classes, [n_sampled_images, 1])
        t_temp1, t_temp2, t_cooling1, t_cooling2, t_ht, t_ft, t_mag = class_maker(batch_size=test_labels.size(0), labels=test_labels, class_table=class_table)
        test_labels = test_labels.to(device)
        test_labels = test_labels.unsqueeze(1).long()
        t_temp1 = t_temp1.long().to(device)
        t_temp2 = t_temp2.long().to(device)
        t_cooling1 = t_cooling1.long().to(device)
        t_cooling2 = t_cooling2.long().to(device)
        t_ht = t_ht.long().to(device)
        t_ft = t_ft.long().to(device)
        t_mag = t_mag.long().to(device)

        ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, t_temp1, t_temp2, t_cooling1, t_cooling2, t_ht, t_ft, t_mag, cfg_scale=0)
        ema_sampled_images = reverse_transforms(ema_sampled_images)

        show_predict_images(ema_sampled_images, test_labels, label_dict)
        exit()

# 根据参数生成图片（抗拉强度或延伸率） para:'UTS'或'Elongation' value为值
def predictByPara(para, value):
    with torch.no_grad():
        result = find_closest_match(df, value, para)
        print(result)
        print(round(result[4], -1), result[6])
        filename = str(round(result[4], -1)) + "-" + str(result[6]) + get_string_timestamp() + '.jpg'
        files= [filename]
        idx = get_label_idx(label_dict, round(result[4], -1), round(result[6], -1))

        if idx < 0:
            return
        print(idx)
        test_labels = [[idx]]
        t_temp1, t_temp2, t_cooling1, t_cooling2, t_ht, t_ft, t_mag = class_maker(batch_size=test_labels.size(0), labels=test_labels, class_table=class_table)
        test_labels = test_labels.to(device)
        test_labels = test_labels.unsqueeze(1).long()
        t_temp1 = t_temp1.long().to(device)
        t_temp2 = t_temp2.long().to(device)
        t_cooling1 = t_cooling1.long().to(device)
        t_cooling2 = t_cooling2.long().to(device)
        t_ht = t_ht.long().to(device)
        t_ft = t_ft.long().to(device)
        t_mag = t_mag.long().to(device)

        ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, t_temp1, t_temp2, t_cooling1, t_cooling2, t_ht, t_ft, t_mag, cfg_scale=0)
        ema_sampled_images = reverse_transforms(ema_sampled_images)

        show_predict_images(ema_sampled_images, test_labels, label_dict, files)


#损失函数
mse = nn.MSELoss()

#创建模型
model = UNet_conditional(num_classes=num_classes).to(device)
ema_model = UNet_conditional(num_classes=num_classes).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
diffusion = Diffusion(img_size=image_size)

#如果是生成设置is_predict = True
if is_predict:
    load_model(load_dir)
    #predictByPara('UTS', 1056.985)

    round =10
    #一轮生成四张，可修改下面的
    for i in range(round):
        predict()
    exit()

l = len(train_loader)
ema = EMA(0.995)
ema_model = copy.deepcopy(model).eval().requires_grad_(False)

#modules.load_model(load_dir)

print('Start training...')
best_loss = float('inf')  # 初始化最佳损失为正无穷大
for e in range(1, n_epoch+1):
    loss_epoch = 0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = aug_transform(images)
        images = images.to(device)
        temp1, temp2, cooling1, cooling2, ht, ft, mag = class_maker(batch_size=labels.size(0), labels=labels, class_table=class_table)
        temp1 = temp1.long().to(device)
        temp2 = temp2.long().to(device)
        cooling1 = cooling1.long().to(device)
        cooling2 = cooling2.long().to(device)
        ht = ht.long().to(device)
        ft = ft.long().to(device)
        mag = mag.long().to(device)
        labels = labels.long().to(device)
        t = diffusion.sample_timesteps(images.shape[0]).to(device)
        x_t, noise = diffusion.noise_images(images, t)

        predicted_noise = model(x_t, t, temp1, temp2, cooling1, cooling2, ht, ft, mag)
        loss = mse(noise, predicted_noise)

        #print(loss)
        
        loss.backward()
        optimizer.step()
        ema.step_ema(ema_model, model)
        loss_epoch += loss.item()
    
    print('Epoch: [%d/%d]: Loss: %.3f' %((e), n_epoch, loss_epoch))

    # 如果当前损失更低，则保存模型
    if loss_epoch < best_loss:
        best_loss = loss_epoch
        #save_dir = load_dir #str(current_dir) + '/model/All_CDDM_HR_Cat_F_6.pth.tar'
        save_model(save_dir)
        #torch.save(model.state_dict(), save_dir)
    
    #if e % n_ax == 0:
    if 1 == 1:
        test_labels = torch.randint(0, num_classes, [n_sampled_images, 1])
        t_temp1, t_temp2, t_cooling1, t_cooling2, t_ht, t_ft, t_mag = class_maker(batch_size=test_labels.size(0), labels=test_labels, class_table=class_table)
        test_labels = test_labels.to(device)
        test_labels = test_labels.unsqueeze(1).long()
        t_temp1 = t_temp1.long().to(device)
        t_temp2 = t_temp2.long().to(device)
        t_cooling1 = t_cooling1.long().to(device)
        t_cooling2 = t_cooling2.long().to(device)
        t_ht = t_ht.long().to(device)
        t_ft = t_ft.long().to(device)
        t_mag = t_mag.long().to(device)

        ema_sampled_images = diffusion.sample(ema_model, n_sampled_images, t_temp1, t_temp2, t_cooling1, t_cooling2, t_ht, t_ft, t_mag, cfg_scale=0)
        ema_sampled_images = reverse_transforms(ema_sampled_images)

        show_grids(ema_sampled_images, test_labels, e, label_dict)
        #save_dir = str(current_dir) + '/model/All_CDDM_HR_Cat_F_6.pth.tar'
        save_model(save_dir)

print(torch.cuda.memory_summary(device=device))
