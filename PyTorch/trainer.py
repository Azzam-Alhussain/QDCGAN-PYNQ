import os
import time 
import random
import itertools
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import DataLoader, ConcatDataset 

from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, LSUN

from model import *
from logger import *
from dataset import ImageDataset
from utils import calculate_frechet_distance

class DCGAN_trainer(object):
    def __init__(self, config):
        self.config = config
        # setting the random seed if given
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
            torch.cuda.manual_seed_all(config.random_seed)
        # setting up the directory name for the experiments e.g. W4A4.....    
        prec_name = "{}_W{}A{}".format(config.dataset, config.weight_bitwidth, config.activation_bitwidth)
        experiment_name = "{}_{}".format(prec_name, datetime.now().strftime("%Y%m%d"))
        self.output_dir_path = os.path.join(config.experiments, experiment_name)
        # resuming the experiment from the given path
        if config.resume:
            self.output_dir_path, _ = os.path.split(config.resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)
        # some important directory paths
        self.checkpoints_dir_path = os.path.join(self.output_dir_path, "checkpoints")
        self.random_results_path = os.path.join(self.output_dir_path, "random_results")
        self.fixed_results_path = os.path.join(self.output_dir_path, "fixed_resutls")
        self.export_path = "export"
        # creating some important directories
        if not config.dry_run:
            os.makedirs(self.checkpoints_dir_path, exist_ok=True)
            # os.makedirs(self.fixed_results_path, exist_ok=True)
            os.makedirs(self.random_results_path, exist_ok=True)
            os.makedirs("test_image", exist_ok=True) # for hls csim

        # adding logger
        self.logger = Logger(self.output_dir_path, config.dry_run)
        self.starting_epoch = 1
        
        # ---------------------------------- Datasets ------------------------------------------------
        # defining datasets
        if config.dataset in ("MNIST", "FashionMNIST"):
            self.dim = 32
            # resizing the images to 64x64 and normalizing ti [-1,1]
            transform_train = transforms.Compose([transforms.Resize(self.dim),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5,), (0.5,))])
            builder = MNIST if config.dataset == "MNIST" else FashionMNIST
            self.in_channels = 1
            # concatenating training and test datasets
            train_set = builder(root="./data", train=True, download=True, transform=transform_train)
            test_set = builder(root="./data", train=False, download=True, transform=transform_train)
            dataset = ConcatDataset([train_set, test_set])
        elif config.dataset == "LSUN":
            self.dim = 32
            self.in_channels = 3
            # resizing the images to 64x64 and normalizing ti [-1,1]
            transform_train = transforms.Compose([transforms.Resize(self.dim),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
            dataset = LSUN(root="./data/lsun/", classes='train', transform=transform_train)
        elif config.dataset == "celebA":
            self.dim = 64
            self.in_channels = 3
            path = "./data/" + config.dataset+ "/img_align_celeba/"
            dataset = ImageDataset(path, (self.dim, self.dim))

        else:
            raise Exception("Dataset {} not supported.".format(config.dataset))

        # dataloader
        self.train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                                       pin_memory=False, num_workers=config.num_workers)


        # ---------------------------------- Device ------------------------------------------------
        # setting up the GPU if we are running on a GPU
        if config.gpus is not None:
            config.gpus = [int(i) for i in config.gpus.split(',')]
            self.device = 'cuda:' + str(config.gpus[0])
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)
        

        # ---------------------------------- Model ------------------------------------------------ 
        discriminator = DCGAN_disc(self.in_channels, config).to(self.device)
        generator = DCGAN_gen(self.in_channels, config).to(self.device)
        # moving the model to device e.g. cpu or gpu
        discriminator = discriminator.to(self.device)
        generator = generator.to(self.device)
        # if we are training on more than 1 gpus, parallelize the model across multiple gpus
        if config.gpus is not None and len(config.gpus) > 1:
            discriminator = nn.DataParallel(discriminator, config.gpus)
            generator = nn.DataParallel(generator, config.gpus)
        self.discriminator = discriminator
        self.generator = generator
        # initializing the weights on the DCGAN
        self.discriminator.init_weights(mean=0.0, std=0.02)
        self.generator.init_weights(mean=0.0, std=0.02)

        # ---------------------------------- Loss/optimizers ------------------------------------------------
        # binary cross entropy loss
        # criterion = nn.BCELoss()
        # self.criterion = criterion.to(self.device)
        # RMSprop optimizters for the generator and discriminator
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        self.gen_opt = optim.Adam(self.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
        # learning rate scheduler, None for now, maybe used for CelebA?
        self.scheduler = None

        # ---------------------------------- Resume ------------------------------------------------
        # Resuming the model if given
        if config.resume:
            # replace print with logging
            self.logger.log.info("Loading model checkpoint at: {}".format(config.resume))
            package = torch.load(config.resume, map_location=self.device)
            disc_state_dict = package["disc_state_dict"]
            gen_state_dict = package["gen_state_dict"]
            self.discriminator.load_state_dict(disc_state_dict, strict=True)
            self.generator.load_state_dict(gen_state_dict, strict=True)
        # resuming the optimizer if we are resuming the training session
        if config.resume and not config.evaluate and not config.export:
            if "disc_opt_state" in package.keys():
                self.disc_opt.load_state_dict(package["disc_opt_state"])
            if "gen_opt_state" in package.keys():
                self.gen_opt.load_state_dict(package["gen_opt_state"])
            if "epoch" in package.keys():
                self.starting_epoch = package["epoch"]
        # resuming the schedular if any
        if config.resume and not config.evaluate and not config.export and self.scheduler is not None:
            self.scheduler.last_epoch = package["epoch"] - 1

        # ---------------------------------- Training Variables ------------------------------------------------
        # some important training variables, no. of input noise channels, 
        # labels for fake images (0) and real images (1)
        self.gen_in_ch = config.gen_ch
        self.y_real = torch.ones(config.batch_size, device=self.device)
        self.y_fake = torch.zeros(config.batch_size, device=self.device)
        self.fixed_z = torch.randn((5*5, self.gen_in_ch), device=self.device)
        # printing all the hyperparamters for the record
        for key in config.keys():
            self.logger.log.info("{} = {}".format(key, config[key]))
   
    # function to save the model at checkpoint
    def checkpoint_best(self, epoch, name):
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.log.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            "disc_state_dict" : self.discriminator.state_dict(),
            "gen_state_dict" : self.generator.state_dict(),

            "disc_opt_state" : self.disc_opt.state_dict(),
            "gen_opt_state" : self.gen_opt.state_dict(),

            "epoch" : epoch + 1,
            "config" : self.config,

            }, best_path)

    # function to train the discriminator
    def train_disc(self, input):
        # clearning the previous accumulated gradient if any
        self.discriminator.zero_grad()
        # passing real images to discriminator
        disc_result_real = self.discriminator(input).squeeze()
        # computing the discriminator loss on real images i.e. comparing it with real labels (i.e. 1)
        # disc_real_loss = self.criterion(disc_result_real, self.y_real)
        # sampling the noise from normal distribution [0,1]
        z = torch.randn((self.config.batch_size, self.gen_in_ch), device=self.device)
        # passing the generated noise to generate fake images
        gen_result = self.generator(z).detach()
        # passing the fake images to the discriminator
        disc_result_fake = self.discriminator(gen_result).squeeze()
        # computing discriminator loss on fake images i.e. comparing it with fake labels (i.e. 0)
        # disc_fake_loss = self.criterion(disc_result_fake, self.y_fake)
        # total loss
        # disc_loss = disc_real_loss + disc_fake_loss
        # adding wasserstein loss to prevent overtraining
        disc_loss = torch.mean(disc_result_fake) - torch.mean(disc_result_real)
        # gradient penelty
        gp = self.compute_gradient_penalty(input, gen_result) 
        disc_loss += 10.0*gp
        # training update step
        disc_loss.backward()
        self.disc_opt.step()
        # clipping the discriminator weights as given in the wasserstein loss paper
        # self.discriminator.clip_weights(-0.01, 0.01)
        return disc_loss.item() 


    # function to compute gradient penalty
    def compute_gradient_penalty(self, real_imgs, fake_imgs):
        alpha = torch.randn((self.config.batch_size, self.in_channels, 1, 1), device=self.device)
        interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
        interpolates_result = self.discriminator(interpolates)
        grad_holder = torch.empty((self.config.batch_size, 1), \
                                   requires_grad=False, device=self.device).fill_(1.0)

        gradients = autograd.grad(outputs=interpolates_result, inputs=interpolates, \
                                  grad_outputs=grad_holder, create_graph=True, retain_graph=True, \
                                  only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


    # function to train the generator
    def train_gen(self, input):
        # clearning the previous accumulated gradient if any
        self.generator.zero_grad()
        # sampling noise and generating the fake images again
        z = torch.randn((self.config.batch_size, self.gen_in_ch), device=self.device)
        gen_result = self.generator(z)
        # passing fake images to discrinomator
        disc_result = self.discriminator(gen_result).squeeze()
        # computing the generator loss on fake images prediction of discriminator and real labels (i.e. 1)
        # gen_loss = self.criterion(disc_result, self.y_real)
        # adding wasserstein loss to prevent overtraining
        gen_loss = -torch.mean(disc_result)
        # training update step        
        gen_loss.backward()
        self.gen_opt.step()
        return gen_loss.item()


    # function to load the data in batches the device (CPU/GPU) and training both discriminator and 
    # generator and evaluating the generator as well
    def train_model(self):
        # iterate through all epochs
        for epoch in range(self.starting_epoch, self.config.epochs+1):
            # turning the train mode
            self.discriminator.train()
            self.generator.train()
            # self.criterion.train()
            # meters to track losses
            epoch_meters = TrainingEpochMeters()
            # iterate over all batches of data
            for i, data in enumerate(self.train_loader):
                (input, _) = data
                # moving input to GPU
                input = input.to(self.device, non_blocking=True)
                # training the models
                disc_loss = self.train_disc(input)
                if i%3 == 0:
                    gen_loss = self.train_gen(input)
                    # clipping the generator weight for quantization
                    self.generator.clip_weights(-1,1)
                    epoch_meters.gen_loss.update(gen_loss)
                    epoch_meters.disc_loss.update(disc_loss)
                # logging
                if i % int(self.config.log_freq) == 0 or i == len(self.train_loader) -1:
                    self.logger.training_batch_cli_log(epoch_meters, epoch, self.config.epochs, i+1, len(self.train_loader))
            # evaluating the generator
            with torch.no_grad():
                p = self.random_results_path + f"/epoch_I{self.config.io_bitwidth}W{self.config.weight_bitwidth}A{self.config.activation_bitwidth}.png"
                self.eval_model(epoch, path=p, isFix=False)
                # fixed_p = self.fixed_results_path + f"/epoch_I{self.config.io_bitwidth}W{self.config.weight_bitwidth}A{self.config.activation_bitwidth}.png"
                # self.eval_model(epoch, path=fixed_p, isFix=True)
            # save the model
            if not self.config.dry_run:
                name = f"checkpoint_I{self.config.io_bitwidth}W{self.config.weight_bitwidth}A{self.config.activation_bitwidth}.tar"
                self.checkpoint_best(epoch, name)

    # function to evaluate model
    def eval_model(self, num_epoch='last_epoch', show=False, path='out.png', isFix=False, examples=5):
        #  switching to the eval mode
        self.generator.eval()
        # passing sampled noise and generating fake images
        z = self.fixed_z if isFix else torch.randn((examples*examples, self.gen_in_ch), device=self.device)
        start = time.time()
        test_images = self.generator(z).cpu()
        end = time.time()
        fps = (examples*examples)/(end-start)
        # quantize the input/output image to Q1.7 format
        z = self.generator.quantize(z)
        test_images = self.generator.quantize(test_images, max_val=(1-2**-6), frac=6)

        # if no. of examples is > 1 then save it as a grid of examples x examples
        if examples > 1:
            size_figure_grid = examples
            cmap = 'gray' if self.config.dataset == "MNIST" else None
            fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(examples,examples))
            for i,j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
                ax[i,j].get_xaxis().set_visible(False)
                ax[i,j].get_yaxis().set_visible(False)

            for k in range(examples*examples):
                i = k//examples
                j = k%examples
                single_image = test_images[k]
                single_image = (((single_image - single_image.min()) * 255) / (single_image.max() - single_image.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
                # single_image = single_image.cpu().data.numpy()#.transpose(1, 2, 0)
                ax[i,j].cla()
                ax[i,j].imshow(single_image, cmap=cmap)
            label = 'Epoch {}'.format(num_epoch)
            fig.text(0.5, 0.04, label, ha='center')

            if not self.config.dry_run:
                plt.savefig(path)
            if show:
                plt.show()
            else:
                plt.close()

        # if no. of examples = 1 save it as a single output png image
        elif examples == 1:
            z = z.cpu().data.numpy().reshape(-1)
            # z = np.concatenate([z, np.zeros(28, dtype=z.dtype)], axis=-1)
            single_image = test_images[0]
            if self.config.dataset != "MNIST":
                single_image = (((single_image - single_image.min()) * 255) / (single_image.max() - single_image.min())).data.numpy().transpose(1, 2, 0).astype(np.uint8)
            else:
                single_image = single_image[0]*255
            from PIL import Image
            from matplotlib import cm
            im = Image.fromarray(np.uint8(single_image))
            im.save(path)
            # just of HLS debugging
            # np.savetxt("test_image/hls/input.txt", z, fmt='%.8f')
            # np.savetxt("test_image/pt_output.txt", single_image.reshape(-1), fmt='%.0f')
        self.logger.log.info("FPS: {}".format(fps))
        self.logger.log.info("Image generated at {}".format(path))


    # exporing the model in evaluation mode
    def export_model(self):
        os.makedirs(self.export_path, exist_ok=True)
        self.generator.eval()
        self.generator.export(self.export_path)


    def calculate_activation_statistics(self, images, model, dims=2048):
        act = np.empty((len(images), dims))        
        images = images.to(self.device, non_blocking=True)
        pred = model(images)[0]
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        act = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


    def calculate_fid(self):
        self.logger.info("Getting the real and fake data...")
        real_data = torch.zeros(self.config.batch_size, 3, self.dim, self.dim)
        real_data[:, :self.in_channels, :, :], _ = next(iter(self.train_loader))

        fake_data = torch.zeros(self.config.batch_size, 3, self.dim, self.dim)
        self.generator.eval()
        z = torch.randn((self.config.batch_size, self.gen_in_ch), device=self.device)
        fake_data[:, :self.in_channels, :, :] = self.generator(z)

        self.logger.info("Running the InceptionV3 model...")
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.model_inception = InceptionV3([block_idx])
        self.model_inception = self.model_inception.to(self.device)
        self.model_inception.eval()
        mu_1, std_1 = self.calculate_activation_statistics(real_data, self.model_inception)
        mu_2, std_2 = self.calculate_activation_statistics(fake_data, self.model_inception)
        
        self.logger.info("Calculating the FID Score...")
        fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        print("FID Score: {:.2f}".format(fid_value))
        return fid_value


