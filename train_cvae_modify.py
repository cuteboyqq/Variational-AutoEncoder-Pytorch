import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn. functional as F
import torch.optim as optim
import os
import traceback
import sys


batch_size = 100
learning_rate = 1e-3
max_epoch = 100
device = torch.device("cuda")
num_workers = 5
load_epoch = 98 #-1
generate = True

class Model(nn.Module):
    def __init__(self,latent_size=32,num_classes=3):
        super(Model,self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes

        # For encode
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, stride=2) #2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.linear1 = nn.Linear(13*13*32,300) #4*4*32
        self.mu = nn.Linear(300, self.latent_size)
        self.logvar = nn.Linear(300, self.latent_size)

        # For decoder
        self.linear2 = nn.Linear(self.latent_size + self.num_classes, 300)
        self.linear3 = nn.Linear(300,13*13*32) #4*4*32
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=5,stride=2)
        self.conv4 = nn.ConvTranspose2d(16, 6, kernel_size=5, stride=2)#16,1
        self.conv5 = nn.ConvTranspose2d(6, 3, kernel_size=4)#1,1

    def encoder(self,x,y):
        y = torch.argmax(y, dim=1).reshape((y.shape[0],1,1,1))
        #print('y:{}'.format(y.shape))
        y = torch.ones(x.shape).to(device)*y
        #print('y:{}'.format(y.shape))
        t = torch.cat((x,y),dim=1)
        #print('x:{}'.format(x.shape))
        #print('t:{}'.format(t.shape))
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = t.reshape((x.shape[0], -1))
        
        t = F.relu(self.linear1(t))
        mu = self.mu(t)
        logvar = self.logvar(t)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std).to(device)
        return eps*std + mu
    
    def unFlatten(self, x):
        return x.reshape((x.shape[0], 32, 13, 13))#32,4,4

    def decoder(self, z):
        t = F.relu(self.linear2(z))
        t = F.relu(self.linear3(t))
        t = self.unFlatten(t)
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = F.relu(self.conv5(t))
        return t


    def forward(self, x, y):
        mu, logvar = self.encoder(x,y)
        #print('mu: {}'.format(mu.shape))
        #print('logvar: {}'.format(logvar.shape))
        z = self.reparameterize(mu,logvar)

        # Class conditioning
        #print('y: {}'.format(y.shape))
        #print('z: {}'.format(z.shape))
        z = torch.cat((z, y.float()), dim=1)
        #print('cat z=yz: {}'.format(z.shape))
        pred = self.decoder(z)
        return pred, mu, logvar



def renormalize(tensor):
        minFrom= tensor.min()
        maxFrom= tensor.max()
        minTo = 0
        maxTo=1
        return minTo + (maxTo - minTo) * ((tensor - minFrom) / (maxFrom - minFrom))


def plot(epoch, pred, y,name='test_'):
    class_id = ['line', 'noline', 'others', 'unkown1', 'unknown2']
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    fig = plt.figure(figsize=(32,32))
    for i in range(1): #6
        #pred[i] = pred[i][:,:,::-1].transpose((2,1,0))
        #pred[i] = np.array(pred[i])
        ax = fig.add_subplot(3,2,i+1)
        #pred[i] = renormalize(pred[i])
        ax.imshow(pred[i][0],cmap='gray')#cmap='gray'
        ax.axis('off')
        ax.title.set_text(str(class_id[y[i]]))
    plt.savefig("./images/{}epoch_{}.jpg".format(name, epoch))
    # plt.figure(figsize=(10,10))
    # plt.imsave("./images/pred_{}.jpg".format(epoch), pred[0,0], cmap='gray')
    plt.close()


def loss_function(x, pred, mu, logvar):
    recon_loss = F.mse_loss(pred, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss, kld


def train(epoch, model, train_loader, optim, args):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    for i,(x,y) in enumerate(train_loader):
        try:
            label = np.zeros((x.shape[0], args.ncls))#10
            label[np.arange(x.shape[0]), y] = 1
            label = torch.tensor(label)

            optim.zero_grad()   
            pred, mu, logvar = model(x.to(device),label.to(device))
            
            recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
            loss = recon_loss + kld
            loss.backward()
            optim.step()

            total_loss += loss.cpu().data.numpy()*x.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
            kld_loss += kld.cpu().data.numpy()*x.shape[0]
            if i == 0:
                print("Gradients")
                for name,param in model.named_parameters():
                    if "bias" in name:
                        print(name,param.grad[0],end=" ")
                    else:
                        print(name,param.grad[0,0],end=" ")
                    print()
        except Exception as e:
            #print('Exception:{}'.format(e))
            traceback.print_exe()
            torch.cuda.empty_cache()
            continue
            
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss

import cv2
def test(epoch, model, test_loader, args):
    count = 1
    save_img = True
    save_dir = "/home/ali/CVAE_MNIST/runs/detect/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    with torch.no_grad():
        for i,(x,y) in enumerate(test_loader):
            try:
                label = np.zeros((x.shape[0], args.ncls)) #10
                label[np.arange(x.shape[0]), y] = 1
                label = torch.tensor(label)

                pred, mu, logvar = model(x.to(device),label.to(device))
                recon_loss, kld = loss_function(x.to(device),pred, mu, logvar)
                loss = recon_loss + kld

                total_loss += loss.cpu().data.numpy()*x.shape[0]
                reconstruction_loss += recon_loss.cpu().data.numpy()*x.shape[0]
                kld_loss += kld.cpu().data.numpy()*x.shape[0]
                if i == 0:
                    print('nothing')
                    # print("gr:", x[0,0,:5,:5])
                    # print("pred:", pred[0,0,:5,:5])
                    #pred = pred[:,:,::-1].transpose((2,1,0))
                    #plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy())
                if save_img:
                    file_name = str(count) + '.jpg'
                    pred.cpu().data.numpy()
                    pred = pred[:,:,::-1].transpose((2,1,0))
                    save_path = os.path.join(save_dir,file_name)
                    cv2.imwrite(save_path,pred)
                    count+=1
            except Exception as e:
                traceback.print_exe()
                #print('Exception:{}'.format(e))
                torch.cuda.empty_cache()
                continue
    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss        



def generate_image(epoch,z, y, model, args):
    with torch.no_grad():
        label = np.zeros((y.shape[0], args.ncls))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)

        pred = model.decoder(torch.cat((z.to(device),label.float().to(device)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy(),name='Eval_')
        print("data Plotted")



def load_data():
    transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=transform),batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    return train_loader, test_loader

def save_model(model, epoch):
    if not os.path.isdir("./checkpoints"):
        os.mkdir("./checkpoints")
    file_name = './checkpoints/model_{}.pt'.format(epoch)
    torch.save(model.state_dict(), file_name)


def get_args():
    import argparse
    #isize=64, nz=100, nc=3
    parser = argparse.ArgumentParser()
    #'/home/ali/datasets/train_video/NewYork_train/train/images'
    parser.add_argument('-imgdir','--img-dir',help='train image dir',default=r"/home/ali/YOLOV5/runs/detect/f_384_2min/crops_ori")
    parser.add_argument('-imgdirtest','--img-dirtest',help='test image dir',default=r"/home/ali/YOLOV5/runs/detect/f_384_2min/crops_ori")
    parser.add_argument('-imgsize','--img-size',type=int,help='image size',default=64)
    parser.add_argument('-nz','--nz',type=int,help='compress length',default=200)
    parser.add_argument('-nc','--nc',type=int,help='num of channel',default=3)
    parser.add_argument('-lr','--lr',type=float,help='learning rate',default=2e-4)
    parser.add_argument('-batchsize','--batch-size',type=int,help='train batch size',default=1)
    parser.add_argument('-savedir','--save-dir',help='save model dir',default=r"/home/ali/AutoEncoder-Pytorch/runs/train")
    parser.add_argument('-weights','--weights',help='save model dir',default='')
    parser.add_argument('-epoch','--epoch',type=int,help='num of epochs',default=30)
    parser.add_argument('-train','--train',type=bool,help='train model',default=True)
    parser.add_argument('-ncls','--ncls',type=int,help='num of class',default=5)
    return parser.parse_args()    



import torchvision.transforms as transforms
def load_data2(img_dir,args):
    size = (args.img_size,args.img_size)
    img_data = torchvision.datasets.ImageFolder(img_dir,
                                                transform=transforms.Compose([
                                                transforms.Resize(size),
                                                #transforms.RandomHorizontalFlip(),
                                                #transforms.Scale(64),
                                                transforms.CenterCrop(size),                                                 
                                                transforms.ToTensor(),
                                                #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #GANomaly parameter
                                                ])
                                                )
    data_loader = torch.utils.data.DataLoader(img_data, batch_size=args.batch_size,shuffle=True,drop_last=True)
    print('data_loader length : {}'.format(len(data_loader)))
    return data_loader

def generate_image_2(epoch,z, y, model, args):
    with torch.no_grad():
        label = np.zeros((y.shape[0], args.ncls))
        label[np.arange(z.shape[0]), y] = 1
        label = torch.tensor(label)

        pred = model.decoder(torch.cat((z.to(device),label.float().to(device)), dim=1))
        plot(epoch, pred.cpu().data.numpy(), y.cpu().data.numpy(),name='Eval_')
        print("data Plotted")

if __name__ == "__main__":
    
    TRAIN = False
    TEST = True
    args = get_args()
    
    train_loader =  load_data2(args.img_dir,args)
    test_loader =  load_data2(args.img_dirtest,args)
    
    #train_loader, test_loader = load_data()
    print("dataloader created")
    model = Model(num_classes=args.ncls).to(device)
    print("model created")
    
    if load_epoch > 0:
        model.load_state_dict(torch.load('./checkpoints/model_{}.pt'.format(load_epoch), map_location=torch.device('cpu')))
        print("model {} loaded".format(load_epoch))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    if TRAIN:
        train_loss_list = []
        test_loss_list = []
        for i in range(load_epoch+1, max_epoch):
            model.train()
            train_total, train_kld, train_loss = train(i, model, train_loader, optimizer, args)
            with torch.no_grad():
                model.eval()
                test_total, test_kld, test_loss = test(i, model, test_loader, args)
                if generate:
                    z = torch.randn(6, 32).to(device)
                    y = torch.tensor([1,1,2,2,4,5]) - 1
                    #y = torch.tensor([1,2,3]) - 1
                    generate_image(i,z, y, model, args)
                
            print("Epoch: {}/{} Train loss: {}, Train KLD: {}, Train Reconstruction Loss:{}".format(i, max_epoch,train_total, train_kld, train_loss))
            print("Epoch: {}/{} Test loss: {}, Test KLD: {}, Test Reconstruction Loss:{}".format(i, max_epoch, test_loss, test_kld, test_loss))
    
            save_model(model, i)
            train_loss_list.append([train_total, train_kld, train_loss])
            test_loss_list.append([test_total, test_kld, test_loss])
            np.save("train_loss", np.array(train_loss_list))
            np.save("test_loss", np.array(test_loss_list))
  
    if TEST:
        for i in range(load_epoch+1, max_epoch):
            with torch.no_grad():
                model.eval()
                test_total, test_kld, test_loss = test(i, model, test_loader, args)
                #generate_image_2(i,z, y, model, args)
    # i, (example_data, exaple_target) = next(enumerate(test_loader))
    # print(example_data[0,0].shape)
    # plt.figure(figsize=(5,5), dpi=100)
    # plt.imsave("example.jpg", example_data[0,0], cmap='gray',  dpi=1000)
    
