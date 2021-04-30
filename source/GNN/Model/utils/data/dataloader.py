from torch.utils.data import DataLoader
class SantanderDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(SantanderDataloader, self).__init__(*args, **kwargs)