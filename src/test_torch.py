import torch
print(torch.__version__)
print(torch.__file__)

def runtest():
  x = torch.ones(100,100)
  x = x.cuda()
  y = torch.ones(100,100)
  y = y.cuda()
  print(x+y)

if __name__ == '__main__':
  runtest()