# python native
import os 
import json
import random
import datetime
import argparse

# external library
import cv2
import numpy as np 
import albumentations as A
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold

# torch
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 


CLASSES = ['finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',]

CLS2IND = {k:v for v, k in enumerate(CLASSES)}
IND2CLS = {v:k for k, v in CLS2IND.items()}


def set_seed(cfg) -> None:
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    
def init_data(cfg) -> set:
    # load files
    dcm_dir = os.path.join(cfg.data_dir, 'DCM')
    
    pngs = {
        os.path.relpath(os.path.join(root, filename), start=dcm_dir)
        for root, _dirs, filenames in os.walk(dcm_dir)
        for filename in filenames
        if os.path.splitext(filename)[1].lower() == '.png'
    }
    
    # convert set to sorted list
    pngs = sorted(pngs)
    
    print(f"# of data: {len(pngs)}")
    
    return pngs

def encode_mask_to_rle(mask):
    '''
    mask: numpy array binary mask 
    1 - mask 
    0 - background
    Returns encoded run length 
    '''
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def decode_rle_to_mask(rle, height, width):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(height * width, dtype=np.uint8)
    
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    
    return img.reshape(height, width)

class XrayDataset(Dataset):
    def __init__(self, 
                 cfg, pngs, jsons, 
                 is_train=True, transforms=None) -> None:
        _filenames = np.array(sorted(pngs))

        self.cfg = cfg
        self.filenames = _filenames
        self.transforms = transforms
        
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor):
        dcm_dir = os.path.join(self.cfg.data_dir, 'DCM')

        image_name = self.filenames[index]
        image_path = os.path.join(dcm_dir, image_name)

        # image, (H, W, C), numpy
        image = cv2.imread(image_path)
        image = image / 255.
        
        if self.transforms:
            inputs = {"image": image}
            result = self.transforms(**inputs)
            image = result['image']

        image = image.transpose(2, 0, 1)
        
        image = torch.from_numpy(image).float()
            
        return image, image_name


pool_scores = []

def hook_fn_save_pool_scores(module, input, output):
    global pool_scores 
    pool_scores.append(output)
    
class FCN8s(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        
        # make feature extractor
        self.features = _make_layers()
        
        # add hooks        
        self.features.pool3.register_forward_hook(hook_fn_save_pool_scores)
        self.features.pool4.register_forward_hook(hook_fn_save_pool_scores)
        
        # pool score
        self.score_pool3 = nn.Conv2d(256, len(CLASSES), 1, 1, 0)
        self.score_pool4 = nn.Conv2d(512, len(CLASSES), 1, 1, 0)
        
        # up sampling
        self.upscore2_score = nn.ConvTranspose2d(len(CLASSES), len(CLASSES), 4, 2, 1)
        self.upscore2_pool4 = nn.ConvTranspose2d(len(CLASSES), len(CLASSES), 4, 2, 1)
        self.upscore8_final = nn.ConvTranspose2d(len(CLASSES), len(CLASSES), 16, 8, 4)
                
    def forward(self, x):
        h = self.features(x)
        score_pool4 = self.score_pool4(pool_scores.pop())
        score_pool3 = self.score_pool3(pool_scores.pop())
        
        upscore = self.upscore2_score(h)
        h = upscore + score_pool4

        upscore_pool4 = self.upscore2_pool4(h)
        h = upscore_pool4 + score_pool3

        upscore_final = self.upscore8_final(h)
        
        return upscore_final
        
        
def _make_layers() -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    
    # 1st convolution, (224, 224, 3) -> (224, 224, 64)
    in_channels = 3
    out_channels = 64
    for i in range(2):
        layers.add_module(f'conv1_{i+1}', nn.Conv2d(in_channels, out_channels, 3, (1,1), (1,1)))
        layers.add_module(f'relu1_{i+1}', nn.ReLU(True))
        in_channels = out_channels
    # 1st pooling, (224, 224, 64) -> (112, 112, 64)
    layers.add_module(f'pool1', nn.MaxPool2d((2,2), (2,2)))
    
    # 2nd convolution, (112, 112, 64) -> (112, 112, 128)
    out_channels *= 2
    for i in range(2):
        layers.add_module(f'conv2_{i+1}', nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.add_module(f'relu2_{i+1}', nn.ReLU(True))
        in_channels = out_channels
    # 2nd pooling, (112, 112, 128) -> (56, 56, 128)
    layers.add_module(f'pool2', nn.MaxPool2d((2,2), (2,2)))
    
    # 3rd convolution, (56, 56, 128) -> (56, 56, 256)
    out_channels *= 2
    for i in range(3):
        layers.add_module(f'conv3_{i+1}', nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.add_module(f'relu3_{i+1}', nn.ReLU(True))
        in_channels = out_channels
    # 3rd pooling, (56, 56, 256) -> (28, 28, 256) --(hook)-> SAVE!
    layers.add_module(f'pool3', nn.MaxPool2d((2,2), (2,2)))
    
    # 4th convolution, (28, 28, 256) -> (28, 28, 512)
    out_channels *= 2
    for i in range(3):
        layers.add_module(f'conv4_{i+1}', nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.add_module(f'relu4_{i+1}', nn.ReLU(True))
        in_channels = out_channels
    # 4th pooling, (28, 28, 512) -> (14, 14, 512) --(hook)-> SAVE!
    layers.add_module(f'pool4', nn.MaxPool2d((2,2), (2,2)))
    
    # 5th convolution, (14, 14, 512) -> (14, 14, 512)
    for i in range(3):
        layers.add_module(f'conv5_{i+1}', nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        layers.add_module(f'relu5_{i+1}', nn.ReLU(True))
        in_channels = out_channels
    # 5th pooling, (14, 14, 512) -> (7, 7, 512)
    layers.add_module(f'pool5', nn.MaxPool2d((2,2), (2,2)))
    
    # Fully Convolutional Layer, (7, 7, 512) -> (7, 7, NCLS)
    layers.add_module(f'fc6', nn.Conv2d(in_channels, 4096, 1, 1, 0))
    layers.add_module(f'relu6', nn.ReLU(True))
    layers.add_module(f'drop6', nn.Dropout2d())
    layers.add_module(f'fc7', nn.Conv2d(4096, 4096, 1, 1, 0))
    layers.add_module(f'relu7', nn.ReLU(True))
    layers.add_module(f'drop7', nn.Dropout2d())
    layers.add_module(f'score', nn.Conv2d(4096, len(CLASSES), 1, 1, 0))
        
    return layers


def train_one_epoch(dataloader, is_train, model, optim, criterion, device):
    # switch mode
    if is_train:
        model.train()
    else:
        model.eval()
        
    model = model.to(device)
    avg_loss = 0
        
    for step, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)

        output_h, output_w = outputs.size(-2), outputs.size(-1)
        mask_h, mask_w = labels.size(-2), labels.size(-1)
        
        # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
        if output_h != mask_h or output_w != mask_w:
            outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")

        loss = criterion(outputs, labels)
        
        if is_train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        
        if (step + 1) % 25 == 0:
            print(
                f'\t{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                f'Step [{step+1}/{len(dataloader)}], '
                f'Loss: {round(loss.item(), 4)}'
            )
        
        avg_loss += loss.item()
    
    return avg_loss / len(dataloader)

def save_model(cfg, model, file_name='last.pth'):
    output_path = os.path.join(cfg.output_dir, file_name)
    torch.save(model, output_path)

def main(cfg) -> None:
    # set seed
    set_seed(cfg)
    
    # init data
    pngs, jsons = init_data(cfg)
    
    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    # transform
    tf = A.Compose(
        A.Resize(224, 224)
    )
    
    # dataset
    train = XrayDataset(cfg, pngs, jsons, is_train=True, transforms=tf)
    valid = XrayDataset(cfg, pngs, jsons, is_train=False, transforms=tf)

    # dataLoader
    train_dataloader = DataLoader(train, 
                                  batch_size=cfg.batch_size, 
                                  shuffle=True,
                                  num_workers=8,
                                  drop_last=True)
    valid_dataloader = DataLoader(valid, 
                                  batch_size=4,     # reduce to 4, because of memory error
                                  shuffle=False,
                                  num_workers=8,    # reduce batch_size but raise num_workers
                                  drop_last=False)

    # model
    model = FCN8s(cfg)
    
    # loss, optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=1e-6)
    
    # train
    logs = {'train': {'loss': []}, 'valid': {'loss': []} }
    for i in range(cfg.epoch):
        print(f"Start training #{i:02d}.. \t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        avg_loss = train_one_epoch(dataloader=train_dataloader,
                               is_train=True,
                               model=model, 
                               optim=optimizer, 
                               criterion=criterion,
                               device=device)
        logs['train']['loss'].append(avg_loss)
        print(
            f'\t{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
            f'Epoch [{i+1}/{cfg.epoch}], '
            f'Train Average Loss: {round(avg_loss, 4)}'
        )
        
        # validation    
        print(f"Start validation #{i:02d}.. \t{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        avg_loss = train_one_epoch(dataloader=valid_dataloader,
                            is_train=False,
                            model=model, 
                            optim=None, 
                            criterion=criterion,
                            device=device)
        logs['valid']['loss'].append(avg_loss)
        print(
            f'\t{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
            f'Valid Average Loss: {round(avg_loss, 4)}'
        )

    with open(os.path.join(cfg.output_dir, 'logs.json'), 'w') as f:
        json.dump(logs, f)
    
    # save checkpoint
    save_model(cfg, model, file_name='last_lr1e-3.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/train", type=str)
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--epoch", default=20, type=int)
    
    args = parser.parse_args()
    
    main(args)
