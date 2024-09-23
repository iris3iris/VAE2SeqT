
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms, datasets
from torchvision.utils import save_image

from tqdm import tqdm
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):  # 定义VAE模型
    def __init__(self, img_size, latent_dim):  # 初始化方法
        super(VAE, self).__init__()  # 继承初始化方法
        self.in_channel, self.img_h, self.img_w = img_size  # 由输入图片形状得到图片通道数C、图片高度H、图片宽度W
        self.h = self.img_h // 32  
        self.w = self.img_w // 32 
        hw = self.h * self.w  # 最终特征层的尺寸hxw
        self.latent_dim = latent_dim  # 采样变量Z的长度
        self.hidden_dims = [32, 64, 128, 256, 512]  # 特征层通道数列表
        # 开始构建编码器Encoder
        layers = []  # 用于存放模型结构
        for hidden_dim in self.hidden_dims:  # 循环特征层通道数列表
            layers += [nn.Conv2d(self.in_channel, hidden_dim, 3, 2, 1),  # 添加conv
                       nn.BatchNorm2d(hidden_dim),  # 添加bn
                       nn.LeakyReLU()]  # 添加leakyrelu
            self.in_channel = hidden_dim  # 将下次循环的输入通道数设为本次循环的输出通道数

        self.encoder = nn.Sequential(*layers)  # 解码器Encoder模型结构

        self.fc_mu = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linaer，将特征向量转化为分布均值mu
        self.fc_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linear，将特征向量转化为分布方差的对数log(var)
        # 开始构建解码器Decoder
        layers = []  # 用于存放模型结构
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)  # linaer，将采样变量Z转化为特征向量
        self.hidden_dims.reverse()  # 倒序特征层通道数列表
        for i in range(len(self.hidden_dims) - 1):  # 循环特征层通道数列表
            layers += [nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i + 1], 3, 2, 1, 1),  # 添加transconv
                       nn.BatchNorm2d(self.hidden_dims[i + 1]),  # 添加bn
                       nn.LeakyReLU()]  # 添加leakyrelu
        layers += [nn.ConvTranspose2d(self.hidden_dims[-1], self.hidden_dims[-1], 3, 2, 1, 1),  # 添加transconv
                   nn.BatchNorm2d(self.hidden_dims[-1]),  # 添加bn
                   nn.LeakyReLU(),  # 添加leakyrelu
                   nn.Conv2d(self.hidden_dims[-1], img_size[0], 3, 1, 1),  # 添加conv
                   nn.Tanh()]  # 添加tanh
        self.decoder = nn.Sequential(*layers)  # 编码器Decoder模型结构

    def encode(self, x):  # 定义编码过程
        result = self.encoder(x) 
        result = torch.flatten(result, 1)  
        mu = self.fc_mu(result)  
        log_var = self.fc_var(result)  

        return [mu, log_var]  # 返回分布的均值和方差对数

    def decode(self, z):  # 定义解码过程
        y = self.decoder_input(z).view(-1, self.hidden_dims[0], self.h,
                                       self.w) 
        y = self.decoder(y)  
        return y  # 返回生成样本Y

    def reparameterize(self, mu, log_var):  # 重参数技巧
        std = torch.exp(0.5 * log_var)  # 分布标准差std
        eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
        return mu + eps * std  # 返回对应正态分布中的采样值

    def forward(self, x):  # 前传函数
        mu, log_var = self.encode(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
        z = self.reparameterize(mu, log_var)  # 经过重参数技巧，得到分布采样变量Z
        y = self.decode(z)  # 经过解码过程，得到生成样本Y
        return [y, x, mu, log_var]  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var

    def sample(self, n, cuda):  # 定义生成过程
        z = torch.randn(n, self.latent_dim)  # 从标准正态分布中采样得到n个采样变量Z，长度为latent_dim
        if cuda:  # 如果使用cuda
            z = z.cuda()  # 将采样变量Z加载到GPU
        images = self.decode(z)  # 经过解码过程，得到生成样本Y
        return images  # 返回生成样本Y


def loss_fn(y, x, mu, log_var):  # 定义损失函数
    recons_loss = F.mse_loss(y, x)  # 重建损失，MSE
    # recons_loss = F.l1_loss(y, x)
    kld_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)  # 分布损失，正态分布与标准正态分布的KL散度
    return recons_loss + w * kld_loss  # 最终损失由两部分组成，其中分布损失需要乘上一个系数w


if __name__ == "__main__":
    total_epochs = 500  # epochs
    batch_size = 64  # batch size
    lr = 1e-3  # lr  5e-4
    w = 0.00025  # kld_loss的系数w
    v = 1000 # rec_loss的系数v
    num_workers = 8  # 数据加载线程数
    image_size = 128  # 图片尺寸
    image_channel = 1  # 图片通道
    latent_dim = 32  # 采样变量Z长度
    local_dataset_dir = './train_VAE'
    os.makedirs(sample_images_dir, exist_ok=True)  # 创建生成样本示例存放路径
    os.makedirs(train_dataset_dir, exist_ok=True)  # 创建训练样本存放路径
    cuda = True if torch.cuda.is_available() else False  # 如果cuda可用，则使用cuda
    img_size = (image_channel, image_size, image_size)  # 输入样本形状(1,32,32)

    vae = VAE(img_size, latent_dim)  # 实例化VAE模型，传入输入样本形状与采样变量长度
    if cuda:  # 如果使用cuda
        vae = vae.cuda()  # 将模型加载到GPU
    # dataset and dataloader
    transform = transforms.Compose(  # 图片预处理方法
        [transforms.Resize(image_size),  # 图片resize，(28x28)-->(32,32)
         transforms.Grayscale(num_output_channels=1),
         transforms.ToTensor(),  # 转化为tensor
         transforms.Normalize([0.5], [0.5])]  # 标准化
    )

    dataset = ImageFolder(
        root=local_dataset_dir,
        transform=transform
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )


    # optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)  # 使用Adam优化器
    # ## train loop
    for epoch in range(total_epochs):  # 循环epoch
        total_loss = 0  # 记录总损失
        pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{total_epochs}", postfix=dict,
                    miniters=0.3)  # 设置当前epoch显示进度
        for i, (img, _) in enumerate(dataloader):  # 循环iter
            if cuda:  # 如果使用cuda
                img = img.cuda()  # 将训练数据加载到GPU
            vae.train()  # 模型开始训练
            optimizer.zero_grad()  # 模型清零梯度
            y, x, mu, log_var = vae(img)  # 输入训练样本X，得到生成样本Y，输入样本X，分布均值mu，分布方差对数log_var
            loss = loss_fn(y, x, mu, log_var)  # 计算loss
            loss.backward()  # 反向传播，计算当前梯度
            optimizer.step()  # 根据梯度，更新网络参数
            total_loss += loss.item()  # 累计loss
            pbar.set_postfix(**{"Loss": loss.item()})  # 显示当前iter的loss
            pbar.update(1)  # 步进长度
        pbar.close()  # 关闭当前epoch显示进度
        print("total_loss:%.4f" %
              (total_loss / len(dataloader)))  # 显示当前epoch训练完成后，模型的总损失
        torch.save(vae.state_dict(), './vae_weights_2_17.pth')

    vae.load_state_dict(torch.load('vae_weights_2_17.pth'))
    vae.eval()  # 模型开始验证
    sample_images = vae.sample(25, cuda)  # 获得25个生成样本
    save_image(sample_images.data, "%s/ep%d.png" % (sample_images_dir, (epoch + 1)), nrow=5,
                   normalize=True)  # 保存生成样本示例(5x5)
    sample_images = vae.sample(1, cuda)
    print(sample_images)
    print(sample_images.shape)
    
    


#######################################潜在空间可视化###########################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KernelDensity
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    import torch

    dataset = ImageFolder('./merged_train', transform=transform)

    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    latent_vectors = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            mu, log_var = vae.encode(images)
            # z = vae.reparameterize(mu, log_var)
            latent_vectors.append(mu.cpu().numpy())


    latent_vectors = np.concatenate(latent_vectors, axis=0)


    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.neighbors import KernelDensity
    import seaborn as sns  
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    num_dims = 10  # 选择前十个维度
    latent_vectors_10 = latent_vectors[:, :num_dims]

    latent_df = pd.DataFrame(latent_vectors_10, columns=[f"Dim_{i + 1}" for i in range(num_dims)])


    g = sns.pairplot(latent_df,
                     diag_kind="kde",
                     plot_kws={"color": "red", "s": 8, "marker": "o", "alpha": 0.5},  # 设置散点样式
                     diag_kws={"color": "blue"})


    for ax in g.axes.flatten():
        ax.set_xticks([])  
        ax.set_yticks([])  
        ax.set_xlabel('')  
        ax.set_ylabel('')  
        ax.spines['top'].set_visible(True)  
        ax.spines['right'].set_visible(True)  
        ax.spines['bottom'].set_visible(True)  
        ax.spines['left'].set_visible(True)  

    g.fig.set_size_inches(15, 15)  

    plt.show()







