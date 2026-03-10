import torch
ckpt = torch.load('./model/best.pth', map_location='cpu')
print('Epoch:', ckpt.get('epoch'))
print('Best loss:', ckpt.get('best_loss'))
print('Keys:', list(ckpt.keys()))
