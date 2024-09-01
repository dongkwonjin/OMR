import cv2
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
from libs.utils import *

class Dataset_CULane(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.datalist = load_pickle(f'{cfg.dir["pre0"]}/datalist')
        # self.datalist[0] = 'driver_182_30frame/06010958_0131.MP4/03090'
        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), interpolation=2), transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx, flip=0):
        img = Image.open(f'{self.cfg.dir["dataset"]}/img/{self.datalist[idx]}.jpg').convert('RGB')
        self.org_width, self.org_height = img.size
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.crop((0, self.cfg.crop_size, int(img.size[0]), int(img.size[1])))
        img = self.transform(img)

        return {'img': self.normalize(img),
                'img_rgb': img,
                'org_h': self.org_height, 'org_w': self.org_width}


    def get_label(self, idx, flip=0):
        data = load_pickle(f'{self.cfg.dir["pre0"]}/{self.datalist[idx]}')
        label = cv2.imread(f'{self.cfg.dir["dataset"]}/img/laneseg_label_w16/{self.datalist[idx]}.png', cv2.IMREAD_UNCHANGED)
        if flip == 1:
            label = cv2.flip(label, 1)  # horizontal flip
        label = np.float32(label[self.cfg.crop_size:, :])
        label = cv2.resize(label, dsize=(self.cfg.width, self.cfg.height), interpolation=0)

        return {'label': label,
                'lane_pts':  data['lane_pts_gt']}

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out.update(self.get_image(idx))
        out.update(self.get_label(idx))
        return out

    def __len__(self):
        return len(self.datalist)