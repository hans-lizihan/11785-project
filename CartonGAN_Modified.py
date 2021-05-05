import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
import torchvision.models as tvmodels
from torchvision import datasets, transforms
import torchvision.utils as tvutils
import argparse
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader
import glob
import time

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataloader.py
    batch_size = 8
    num_workers = os.cpu_count()
    photo_image_dir = "dataset/SrcDataSet/"
    animation_image_dir = "dataset/TgtDataSet/Shinkai Makoto/Your Name/"
    edge_smoothed_image_dir = "dataset/TgtDataSet/Shinkai Makoto/Your Name_smooth/"
    test_photo_image_dir = "data/test/"
    num_train_data = 5000

    # CartoonGAN_train.py
    adam_beta1 = 0.5  # following dcgan
    lr = 0.0002
    num_epochs = 100
    initialization_epochs = 10
    content_loss_weight = 10
    print_every = 100

# transforms that will be applied to all datasets
transform = transforms.Compose([
    # resizing and center cropping is not needed since we already did those using preprocessing.py
    transforms.ToTensor(),
    # normalize tensor that each element is in range [-1, 1]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

def load_image_dataloader(root_dir, batch_size=Config.batch_size, num_workers=Config.num_workers, shuffle=True):
    """
    :param root_dir: directory that contains another directory of images. All images should be under root_dir/<some_dir>/
    :param batch_size: batch size
    :param num_workers: number of workers for torch.utils.data.DataLoader
    :param shuffle: use shuffle
    :return: torch.utils.Dataloader object
    """
    assert os.path.isdir(root_dir)

    image_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    dataset = torch.utils.data.Subset(image_dataset, numpy.random.choice(len(image_dataset), min(Config.num_train_data, len(image_dataset)), replace=False))
    dataloader = DataLoader(dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)

    return dataloader


class ResidualBlock(nn.Module):
    def __init__(self, channels=256, use_bias=False):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(channels)
        )

    def forward(self, input):
        residual = input
        x = self.model(input)
        # element-wise sum
        out = x + residual

        return out


class Generator(nn.Module):
    def __init__(self, n_res_block=8, use_bias=False):
        super().__init__()

        # down sampling, or layers before residual blocks
        self.down_sampling = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # res_blocks
        res_blocks = []
        for i in range(n_res_block):
            res_blocks.append(ResidualBlock(channels=256, use_bias=use_bias))
        self.res_blocks = nn.Sequential(*res_blocks)

        # up sapling, or layers after residual blocks
        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=use_bias),
            nn.Tanh()
        )

    def forward(self, input):
        x = self.down_sampling(input)
        x = self.res_blocks(x)
        out = self.up_sampling(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, leaky_relu_negative_slope=0.2, use_bias=False):
        super().__init__()

        self.negative_slope = leaky_relu_negative_slope
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(self.negative_slope, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.negative_slope, inplace=True),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1, bias=use_bias)

        )

    def forward(self, input):
        output = self.layers(input)
        return output


class FeatureExtractor(nn.Module):
    def __init__(self, network='vgg'):
        # in original paper, authors used vgg.
        # however, there exist much better convolutional networks than vgg, and we may experiment with them
        # possible models may be vgg, resnet, etc
        super().__init__()
        assert network in ['vgg']

        if network == 'vgg':
            vgg = tvmodels.vgg19_bn(pretrained=True)
            self.feature_extractor = vgg.features[:37]
            # vgg.features[36] is conv4_4 layer, which is what original CartoonGAN used

        else:
            # TODO
            pass

        # FeatureExtractor should not be trained
        for child in self.feature_extractor.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, input):
        return self.feature_extractor(input)

class CartoonGANTrainer:
    def __init__(self, generator, discriminator, feature_extractor,
                 photo_image_loader, animation_image_loader, edge_smoothed_image_loader,
                 content_loss_weight=Config.content_loss_weight, lsgan=False):
        """

        :param generator: CartoonGAN generator
        :param discriminator: CartoonGAN discriminator
        :param feature_extractor: feature extractor, VGG in CartoonGAN
        :param photo_image_loader:
        :param animation_image_loader:
        :param edge_smoothed_image_loader:
        """

        # just in case our generator and discriminator are not using Config.device
        self.generator = generator.to(Config.device)
        self.discriminator = discriminator.to(Config.device)
        self.feature_extractor = feature_extractor.to(Config.device)

        self.photo_image_loader = photo_image_loader
        self.animation_image_loader = animation_image_loader
        self.edge_smoothed_image_loader = edge_smoothed_image_loader

        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=Config.lr, betas=(Config.adam_beta1, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=Config.lr,
                                         betas=(Config.adam_beta1, 0.999))
        if not lsgan:
            self.disc_criterion = nn.BCEWithLogitsLoss().to(Config.device)  # for discriminator GAN loss
            self.gen_criterion_gan = nn.BCEWithLogitsLoss().to(Config.device)  # for generator GAN loss
        else:
            # use Least Square GAN
            self.disc_criterion = nn.MSELoss().to(Config.device)
            self.gen_criterion_gan = nn.MSELoss().to(Config.device)
        self.gen_criterion_content = nn.L1Loss().to(Config.device)  # for generator content loss
        self.content_loss_weight = content_loss_weight

        self.curr_initialization_epoch = 0
        self.curr_epoch = 0

        self.init_loss_hist = []
        self.loss_D_hist = []
        self.loss_G_hist = []
        self.loss_content_hist = []
        self.print_every = Config.print_every

    def train(self, num_epochs=Config.num_epochs, initialization_epochs=Config.initialization_epochs,
              save_path='checkpoints/CartoonGAN/'):
        # if not initialized, do it!
        if self.curr_initialization_epoch < initialization_epochs:
            for init_epoch in range(self.curr_initialization_epoch, initialization_epochs):
                start = time.time()
                epoch_loss = 0

                for ix, (photo_images, _) in enumerate(self.photo_image_loader, 0):
                    photo_images = photo_images.to(Config.device)

                    loss = self.initialize_step(photo_images)
                    self.init_loss_hist.append(loss)
                    epoch_loss += loss

                    # print progress
                    if (ix + 1) % self.print_every == 0:
                        print("Initialization Phase Epoch {0} Iteration {1}: Content Loss: {2:.4f}".format(init_epoch+1,
                                                                                                           ix + 1,
                                                                                                           epoch_loss / (ix + 1)))

                print("Initialization Phase [{0}/{1}], {2:.4f} seconds".format(init_epoch + 1, initialization_epochs,
                                                                             time.time() - start))
                self.curr_initialization_epoch += 1

        for epoch in range(self.curr_epoch, num_epochs):
            start = time.time()
            epoch_loss_D = 0
            epoch_loss_G = 0
            epoch_loss_content = 0

            for ix, ((animation_images, _), (edge_smoothed_images, _), (photo_images, _)) in enumerate(
                    zip(self.animation_image_loader,
                        self.edge_smoothed_image_loader,
                        self.photo_image_loader), 0):
                # do train_step...!
                animation_images = animation_images.to(Config.device)
                edge_smoothed_images = edge_smoothed_images.to(Config.device)
                photo_images = photo_images.to(Config.device)

                loss_D, loss_G, loss_content = self.train_step(animation_images, edge_smoothed_images, photo_images)
                epoch_loss_D += loss_D
                epoch_loss_G += loss_G
                epoch_loss_content += loss_content

                self.loss_D_hist.append(loss_D)
                self.loss_G_hist.append(loss_G)
                self.loss_content_hist.append(loss_content)

                if (ix + 1) % self.print_every == 0:
                    print("Training Phase Epoch {0} Iteration {1}, loss_D: {2:.4f}, "
                          "loss_G: {3:.4f}, loss_content: {4:.4f}".format(epoch + 1, ix + 1, epoch_loss_D / (ix + 1),
                                                                          epoch_loss_G / (ix + 1),
                                                                          epoch_loss_content / (ix + 1)))

            # end of epoch
            print("Training Phase [{0}/{1}], {2:.4f} seconds".format(epoch + 1, num_epochs, time.time() - start))
            self.curr_epoch += 1

        # Training finished, save checkpoint
        if not os.path.isdir('checkpoints/'):
            os.mkdir('checkpoints/')
        if not os.path.isdir('checkpoints/CartoonGAN/'):
            os.mkdir('checkpoints/CartoonGAN/')

        self.save_checkpoint(os.path.join(save_path, 'checkpoint-epoch-{0}.ckpt'.format(num_epochs)))

        return self.loss_D_hist, self.loss_G_hist, self.loss_content_hist

    def train_step(self, animation_images, edge_smoothed_images, photo_images):
        self.discriminator.zero_grad()
        self.generator.zero_grad()

        loss_D = 0
        loss_G = 0
        loss_content = 0

        # 1. Train Discriminator
        # 1-1. Train Discriminator using animation images
        animation_disc_output = self.discriminator(animation_images)
        animation_target = torch.ones_like(animation_disc_output)
        loss_real = self.disc_criterion(animation_disc_output, animation_target)

        # 1-2. Train Discriminator using edge smoothed images
        edge_smoothed_disc_output = self.discriminator(edge_smoothed_images)
        edge_smoothed_target = torch.zeros_like(edge_smoothed_disc_output)
        loss_edge = self.disc_criterion(edge_smoothed_disc_output, edge_smoothed_target)

        # 1-3. Train Discriminator using generated images
        generated_images = self.generator(photo_images).detach()

        generated_output = self.discriminator(generated_images)
        generated_target = torch.zeros_like(generated_output)
        loss_generated = self.disc_criterion(generated_output, generated_target)

        loss_disc = loss_real + loss_edge + loss_generated

        loss_disc.backward()
        loss_D = loss_disc.item()

        self.disc_optimizer.step()

        # 2. Train Generator
        self.generator.zero_grad()

        # 2-1. Train Generator using adversarial loss, using generated images
        generated_images = self.generator(photo_images)

        generated_output = self.discriminator(generated_images)
        generated_target = torch.ones_like(generated_output)
        loss_adv = self.gen_criterion_gan(generated_output, generated_target)

        # 2-2. Train Generator using content loss
        x_features = self.feature_extractor((photo_images + 1) / 2).detach()
        Gx_features = self.feature_extractor((generated_images + 1) / 2)

        loss_content = self.content_loss_weight * self.gen_criterion_content(Gx_features, x_features)

        loss_gen = loss_adv + loss_content
        loss_gen.backward()

        loss_G = loss_adv.item()
        loss_content = loss_content.item()

        self.gen_optimizer.step()

        return loss_D, loss_G, loss_content

    def initialize_step(self, photo_images):
        self.generator.zero_grad()
        x_features = self.feature_extractor((photo_images + 1) / 2).detach()  # move [-1, 1] to [0, 1]
        Gx = self.generator(photo_images)
        Gx_features = self.feature_extractor((Gx + 1) / 2)  # move [-1, 1] to [0, 1]

        content_loss = self.content_loss_weight * self.gen_criterion_content(Gx_features, x_features)
        content_loss.backward()
        self.gen_optimizer.step()

        return content_loss.item()

    def save_checkpoint(self, checkpoint_path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'curr_epoch': self.curr_epoch,
            'curr_initialization_epoch': self.curr_initialization_epoch,
            'loss_G_hist': self.loss_G_hist,
            'loss_D_hist': self.loss_D_hist,
            'loss_content_hist': self.loss_content_hist
        }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        self.loss_G_hist = checkpoint['loss_G_hist']
        self.loss_D_hist = checkpoint['loss_D_hist']
        self.loss_content_hist = checkpoint['loss_content_hist']
        self.curr_epoch = checkpoint['curr_epoch']
        self.curr_initialization_epoch = checkpoint['curr_initialization_epoch']

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test',
                        action='store_true',
                        help='Use this argument to test generator')

    parser.add_argument('--model_path',
                        help='Path to saved model')

    parser.add_argument('--model_save_path',
                        default='checkpoints/CartoonGAN/',
                        help='Path to save trained model')

    parser.add_argument("--photo_image_dir",
                        default=Config.photo_image_dir,
                        help="Path to photo images")

    parser.add_argument("--animation_image_dir",
                        default=Config.animation_image_dir,
                        help="Path to animation images")

    parser.add_argument("--edge_smoothed_image_dir",
                        default=Config.edge_smoothed_image_dir,
                        help="Path to edge smoothed animation images")

    parser.add_argument('--test_image_path',
                        default=Config.test_photo_image_dir,
                        help='Path to test photo images')

    parser.add_argument('--initialization_epochs',
                        type=int,
                        default=Config.initialization_epochs,
                        help='Number of epochs for initialization phase')

    parser.add_argument('--num_epochs',
                        type=int,
                        default=Config.num_epochs,
                        help='Number of training epochs')

    parser.add_argument("--batch_size",
                        type=int,
                        default=Config.batch_size)

    parser.add_argument('--use_modified_model',
                        action='store_true',
                        help="Use this argument to use modified model")

    args = parser.parse_args()

    return args


def load_model(generator, discriminator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])


def load_generator(generator, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=Config.device)
    generator.load_state_dict(checkpoint['generator_state_dict'])


def generate_and_save_images(generator, test_image_loader, save_path):
    # for each image in test_image_loader, generate image and save
    generator.eval()
    torch_to_image = transforms.Compose([
        transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # [-1, 1] to [0, 1]
        transforms.ToPILImage()
    ])

    image_ix = 0
    with torch.no_grad():
        for test_images, _ in test_image_loader:
            test_images = test_images.to(Config.device)
            generated_images = generator(test_images).detach().cpu()

            for i in range(len(generated_images)):
                image = generated_images[i]
                image = torch_to_image(image)
                image.save(os.path.join(save_path, '{0}.jpg'.format(image_ix)))
                image_ix += 1


def main():

    args = get_args()

    device = Config.device
    print("PyTorch running with device {0}".format(device))

    print("Creating models...")
    generator = Generator().to(device)

    if args.test:
        assert args.model_path, 'model_path must be provided for testing'
        print('Testing...')
        generator.eval()

        print('Loading models...')
        load_generator(generator, args.model_path)

        test_images = load_image_dataloader(root_dir=args.test_image_path, batch_size=1, shuffle=False)

        print("Generating sample images")
        image_batch, _ = next(iter(test_images))
        image_batch = image_batch.to(Config.device)

        with torch.no_grad():
            new_images = generator(image_batch).detach().cpu()

        tvutils.save_image(image_batch, 'test_images.jpg', nrow=4, padding=2, normalize=True, range=(-1, 1))
        tvutils.save_image(new_images, 'generated_images.jpg', nrow=4, padding=2, normalize=True, range=(-1, 1))

        if not os.path.isdir('generated_images/CartoonGAN'):
            os.makedirs('generated_images/CartoonGAN/')

        print("Generating Images")
        # generate new images for all images in args.test_image_path, and save them to generated_images/CartoonGAN/ directory
        generate_and_save_images(generator, test_images, 'generated_images/CartoonGAN/')

    else:
        print("Training...")

        print("Loading Discriminator and Feature Extractor...")
        discriminator = Discriminator().to(device)
        feature_extractor = FeatureExtractor().to(device)

        # load dataloaders
        photo_images = load_image_dataloader(root_dir=args.photo_image_dir, batch_size=args.batch_size)
        animation_images = load_image_dataloader(root_dir=args.animation_image_dir, batch_size=args.batch_size)
        edge_smoothed_images = load_image_dataloader(root_dir=args.edge_smoothed_image_dir, batch_size=args.batch_size)

        print("Loading Trainer...")
        trainer = CartoonGANTrainer(generator, discriminator, feature_extractor, photo_images, animation_images,
                                    edge_smoothed_images, lsgan=args.use_modified_model)
        if args.model_path:
            trainer.load_checkpoint(args.model_path)

        print('Start Training...')
        loss_D_hist, loss_G_hist, loss_content_hist = trainer.train(num_epochs=args.num_epochs,
                                                                    initialization_epochs=args.initialization_epochs,
                                                                    save_path=args.model_save_path)


if __name__ == '__main__':
    main()
