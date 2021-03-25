import torch
import numpy as np


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end
    def __iter__(self):
        return iter(range(self.start, self.end))
# should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].


def test_dataloader():
  ds = MyIterableDataset(start=3, end=7)
  print(list(torch.utils.data.DataLoader(ds, num_workers=0)))


if __name__=='__main__':
  test_dataloader()
