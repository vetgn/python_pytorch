from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

for i in range(100):
    writer.add_scalar("y = x", i, i)

writer.close()
# tensorboard --logdir=logs 打开logs文件
# tensorboard --logdir=logs --port=6007 切换端口
# 想要add其他图像，需将logs中的文件全部删除
