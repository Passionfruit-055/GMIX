from marl import rootpath, exp_name

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(rootpath + exp_name)