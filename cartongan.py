!pip install imageio torch matplotlib numpy torchvision scipy opencv-python tqdm python-box
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import cv2, os, time, pickle
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from box import Box

# ==================
# utils
# ==================
def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1

        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

# =======================
# Edge promoting
# =======================
def edge_promoting(root, save):
    file_list = os.listdir(root)
    if not os.path.isdir(save):
        os.makedirs(save)
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gauss = cv2.getGaussianKernel(kernel_size, 0)
    gauss = gauss * gauss.transpose(1, 0)
    n = 1
    for f in tqdm(file_list):
        rgb_img = cv2.imread(os.path.join(root, f))
        gray_img = cv2.imread(os.path.join(root, f), 0)
        rgb_img = cv2.resize(rgb_img, (256, 256))
        pad_img = np.pad(rgb_img, ((2,2), (2,2), (0,0)), mode='reflect')
        gray_img = cv2.resize(gray_img, (256, 256))
        edges = cv2.Canny(gray_img, 100, 200)
        dilation = cv2.dilate(edges, kernel)

        gauss_img = np.copy(rgb_img)
        idx = np.where(dilation != 0)
        for i in range(np.sum(dilation != 0)):
            gauss_img[idx[0][i], idx[1][i], 0] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 0], gauss))
            gauss_img[idx[0][i], idx[1][i], 1] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 1], gauss))
            gauss_img[idx[0][i], idx[1][i], 2] = np.sum(np.multiply(pad_img[idx[0][i]:idx[0][i] + kernel_size, idx[1][i]:idx[1][i] + kernel_size, 2], gauss))

        result = np.concatenate((rgb_img, gauss_img), 1)

        cv2.imwrite(os.path.join(save, str(n) + '.png'), result)
        n += 1

# ==================
# Layers
# ==================
class ResnetBlock(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(ResnetBlock, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        initialize_weights(self)

    def forward(self, input):
        x = F.relu(self.conv1_norm(self.conv1(input)), True)
        x = self.conv2_norm(self.conv2(x))

        return input + x #Elementwise Sum


class Generator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32, nb=6):
        super(Generator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.nb = nb
        self.down_convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3), #k7n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1), #k3n128s2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.Conv2d(nf * 2, nf * 4, 3, 2, 1), #k3n256s1
            nn.Conv2d(nf * 4, nf * 4, 3, 1, 1), #k3n256s1
            nn.InstanceNorm2d(nf * 4),
            nn.ReLU(True),
        )

        self.resnet_blocks = []
        for i in range(nb):
            self.resnet_blocks.append(ResnetBlock(nf * 4, 3, 1, 1))

        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.up_convs = nn.Sequential(
            nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 1, 1), #k3n128s1/2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1), #k3n128s1
            nn.InstanceNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 3, 2, 1, 1), #k3n64s1/2
            nn.Conv2d(nf, nf, 3, 1, 1), #k3n64s1
            nn.InstanceNorm2d(nf),
            nn.ReLU(True),
            nn.Conv2d(nf, out_nc, 7, 1, 3), #k7n3s1
            nn.Tanh(),
        )

        initialize_weights(self)

    # forward method
    def forward(self, input):
        x = self.down_convs(input)
        x = self.resnet_blocks(x)
        output = self.up_convs(x)

        return output


class Discriminator(nn.Module):
    # initializers
    def __init__(self, in_nc, out_nc, nf=32):
        super(Discriminator, self).__init__()
        self.input_nc = in_nc
        self.output_nc = out_nc
        self.nf = nf
        self.convs = nn.Sequential(
            nn.Conv2d(in_nc, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf * 2, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 2, nf * 4, 3, 1, 1),
            nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 4, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 4, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, nf * 8, 3, 1, 1),
            nn.InstanceNorm2d(nf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf * 8, out_nc, 3, 1, 1),
            nn.Sigmoid(),
        )

        initialize_weights(self)

    # forward method
    def forward(self, input):
        # input = torch.cat((input1, input2), 1)
        output = self.convs(input)

        return output


class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if not init_weights == None:
            self.load_state_dict(torch.load(init_weights))

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x

# ==========
# training
# ==========
args = Box(
    name='project_name',
    src_data='src_data_path',
    tgt_data='tgt_data_path',
    vgg_model='pre_trained_VGG19_model_path/vgg19.pth',
    # input channel for generator
    in_ngc=3,
    # output channel for generator
    out_ngc=3,
    # input channel for discriminator
    in_ndc=3,
    # output channel for discriminator
    out_ndc=1,
    batch_size=8,
    ngf=64,
    ndf=32,
    # number of resblock for generator
    nb=8,
    # size of input
    input_size=256,
    train_epoch=100,
    pre_train_epoch=10,
    lrD=0.0002,
    lrG=0.0002,
    # lambda for content loss
    con_lambda=10,
    # beta1 for adam
    beta1=0.5,
    # beta2 for adam
    beta2=0.999,
    # trained model for generator
    latest_generator_model='',
    # trained model for discriminator
    latest_discriminator_model='',
)

print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# ===============
# Training process
# =====================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    # results save path
    if not os.path.isdir(os.path.join(args.name + '_results', 'Reconstruction')):
        os.makedirs(os.path.join(args.name + '_results', 'Reconstruction'))
    if not os.path.isdir(os.path.join(args.name + '_results', 'Transfer')):
        os.makedirs(os.path.join(args.name + '_results', 'Transfer'))

    # edge-promoting
    if not os.path.isdir(os.path.join('data', args.tgt_data, 'pair')):
        print('edge-promoting start!!')
        edge_promoting(os.path.join('data', args.tgt_data, 'train'), os.path.join('data', args.tgt_data, 'pair'))
    else:
        print('edge-promoting already done')

    # data_loader
    src_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    tgt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_loader_src = utils.data_load(os.path.join('data', args.src_data), 'train', src_transform, args.batch_size, shuffle=True, drop_last=True)
    train_loader_tgt = utils.data_load(os.path.join('data', args.tgt_data), 'pair', tgt_transform, args.batch_size, shuffle=True, drop_last=True)
    test_loader_src = utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)

    # network
    G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
    if args.latest_generator_model != '':
        if torch.cuda.is_available():
            G.load_state_dict(torch.load(args.latest_generator_model))
        else:
            # cpu mode
            G.load_state_dict(torch.load(args.latest_generator_model, map_location=lambda storage, loc: storage))

    D = networks.discriminator(args.in_ndc, args.out_ndc, args.ndf)
    if args.latest_discriminator_model != '':
        if torch.cuda.is_available():
            D.load_state_dict(torch.load(args.latest_discriminator_model))
        else:
            D.load_state_dict(torch.load(args.latest_discriminator_model, map_location=lambda storage, loc: storage))
    VGG = networks.VGG19(init_weights=args.vgg_model, feature_mode=True)
    G.to(device)
    D.to(device)
    VGG.to(device)
    G.train()
    D.train()
    VGG.eval()
    print('---------- Networks initialized -------------')
    utils.print_network(G)
    utils.print_network(D)
    utils.print_network(VGG)
    print('-----------------------------------------------')

    # loss
    BCE_loss = nn.BCELoss().to(device)
    L1_loss = nn.L1Loss().to(device)

    # Adam optimizer
    G_optimizer = optim.Adam(G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
    D_optimizer = optim.Adam(D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))
    G_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=G_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)
    D_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=D_optimizer, milestones=[args.train_epoch // 2, args.train_epoch // 4 * 3], gamma=0.1)

    pre_train_hist = {}
    pre_train_hist['Recon_loss'] = []
    pre_train_hist['per_epoch_time'] = []
    pre_train_hist['total_time'] = []

    """ Pre-train reconstruction """
    if args.latest_generator_model == '':
        print('Pre-training start!')
        start_time = time.time()
        for epoch in range(args.pre_train_epoch):
            epoch_start_time = time.time()
            Recon_losses = []
            for x, _ in train_loader_src:
                x = x.to(device)

                # train generator G
                G_optimizer.zero_grad()

                x_feature = VGG((x + 1) / 2)
                G_ = G(x)
                G_feature = VGG((G_ + 1) / 2)

                Recon_loss = 10 * L1_loss(G_feature, x_feature.detach())
                Recon_losses.append(Recon_loss.item())
                pre_train_hist['Recon_loss'].append(Recon_loss.item())

                Recon_loss.backward()
                G_optimizer.step()

            per_epoch_time = time.time() - epoch_start_time
            pre_train_hist['per_epoch_time'].append(per_epoch_time)
            print('[%d/%d] - time: %.2f, Recon loss: %.3f' % ((epoch + 1), args.pre_train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Recon_losses))))

        total_time = time.time() - start_time
        pre_train_hist['total_time'].append(total_time)
        with open(os.path.join(args.name + '_results',  'pre_train_hist.pkl'), 'wb') as f:
            pickle.dump(pre_train_hist, f)

        with torch.no_grad():
            G.eval()
            for n, (x, _) in enumerate(train_loader_src):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_train_recon_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break

            for n, (x, _) in enumerate(test_loader_src):
                x = x.to(device)
                G_recon = G(x)
                result = torch.cat((x[0], G_recon[0]), 2)
                path = os.path.join(args.name + '_results', 'Reconstruction', args.name + '_test_recon_' + str(n + 1) + '.png')
                plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                if n == 4:
                    break
    else:
        print('Load the latest generator model, no need to pre-train')


    train_hist = {}
    train_hist['Disc_loss'] = []
    train_hist['Gen_loss'] = []
    train_hist['Con_loss'] = []
    train_hist['per_epoch_time'] = []
    train_hist['total_time'] = []
    print('training start!')
    start_time = time.time()
    real = torch.ones(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
    fake = torch.zeros(args.batch_size, 1, args.input_size // 4, args.input_size // 4).to(device)
    for epoch in range(args.train_epoch):
        epoch_start_time = time.time()
        G.train()
        G_scheduler.step()
        D_scheduler.step()
        Disc_losses = []
        Gen_losses = []
        Con_losses = []
        for (x, _), (y, _) in zip(train_loader_src, train_loader_tgt):
            e = y[:, :, :, args.input_size:]
            y = y[:, :, :, :args.input_size]
            x, y, e = x.to(device), y.to(device), e.to(device)

            # train D
            D_optimizer.zero_grad()

            D_real = D(y)
            D_real_loss = BCE_loss(D_real, real)

            G_ = G(x)
            D_fake = D(G_)
            D_fake_loss = BCE_loss(D_fake, fake)

            D_edge = D(e)
            D_edge_loss = BCE_loss(D_edge, fake)

            Disc_loss = D_real_loss + D_fake_loss + D_edge_loss
            Disc_losses.append(Disc_loss.item())
            train_hist['Disc_loss'].append(Disc_loss.item())

            Disc_loss.backward()
            D_optimizer.step()

            # train G
            G_optimizer.zero_grad()

            G_ = G(x)
            D_fake = D(G_)
            D_fake_loss = BCE_loss(D_fake, real)

            x_feature = VGG((x + 1) / 2)
            G_feature = VGG((G_ + 1) / 2)
            Con_loss = args.con_lambda * L1_loss(G_feature, x_feature.detach())

            Gen_loss = D_fake_loss + Con_loss
            Gen_losses.append(D_fake_loss.item())
            train_hist['Gen_loss'].append(D_fake_loss.item())
            Con_losses.append(Con_loss.item())
            train_hist['Con_loss'].append(Con_loss.item())

            Gen_loss.backward()
            G_optimizer.step()


        per_epoch_time = time.time() - epoch_start_time
        train_hist['per_epoch_time'].append(per_epoch_time)
        print(
        '[%d/%d] - time: %.2f, Disc loss: %.3f, Gen loss: %.3f, Con loss: %.3f' % ((epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(Disc_losses)),
            torch.mean(torch.FloatTensor(Gen_losses)), torch.mean(torch.FloatTensor(Con_losses))))

        if epoch % 2 == 1 or epoch == args.train_epoch - 1:
            with torch.no_grad():
                G.eval()
                for n, (x, _) in enumerate(train_loader_src):
                    x = x.to(device)
                    G_recon = G(x)
                    result = torch.cat((x[0], G_recon[0]), 2)
                    path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_train_' + str(n + 1) + '.png')
                    plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                    if n == 4:
                        break

                for n, (x, _) in enumerate(test_loader_src):
                    x = x.to(device)
                    G_recon = G(x)
                    result = torch.cat((x[0], G_recon[0]), 2)
                    path = os.path.join(args.name + '_results', 'Transfer', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
                    plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
                    if n == 4:
                        break

                torch.save(G.state_dict(), os.path.join(args.name + '_results', 'generator_latest.pkl'))
                torch.save(D.state_dict(), os.path.join(args.name + '_results', 'discriminator_latest.pkl'))

    total_time = time.time() - start_time
    train_hist['total_time'].append(total_time)

    print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
    print("Training finish!... save training results")

    torch.save(G.state_dict(), os.path.join(args.name + '_results',  'generator_param.pkl'))
    torch.save(D.state_dict(), os.path.join(args.name + '_results',  'discriminator_param.pkl'))
    with open(os.path.join(args.name + '_results',  'train_hist.pkl'), 'wb') as f:
        pickle.dump(train_hist, f)

args.image_dir = 'image_dir'
args.output_image_dir = 'output_image_dir'
args.pretrained_model = 'pretrained_model'

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = True

    G = networks.generator(args.in_ngc, args.out_ngc, args.ngf, args.nb)
    if torch.cuda.is_available():
        G.load_state_dict(torch.load(args.pre_trained_model))
    else:
        # cpu mode
        G.load_state_dict(torch.load(args.pre_trained_model, map_location=lambda storage, loc: storage))
    G.to(device)

    src_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    # utils.data_load(os.path.join('data', args.src_data), 'test', src_transform, 1, shuffle=True, drop_last=True)
    image_src = utils.data_load(os.path.join(args.image_dir), 'test', src_transform, 1, shuffle=True, drop_last=True)

    with torch.no_grad():
        G.eval()
        for n, (x, _) in enumerate(image_src):
            x = x.to(device)
            G_recon = G(x)
            result = torch.cat((x[0], G_recon[0]), 2)
            path = os.path.join(args.output_image_dir, str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)0
