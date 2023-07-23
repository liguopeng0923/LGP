import matplotlib.pyplot as plt
import os.path as osp
import time
import math

def plt_losses(losses,labels,work_dir):
    plt.xlabel('iteration')
    plt.ylabel('losses')
    k = len(losses)
    for i,l in enumerate(losses):
        xs = []
        ys = []
        for epoch in range(l.shape[0]):
            ys.append((l[epoch]).detach().cpu())
            xs.append(epoch)
        plt.subplot(math.ceil(k/2), math.ceil(k/2), i+1)
        plt.plot(xs, ys,list(labels.values())[i],label = list(labels.keys())[i],linewidth=1)
        plt.title("{}".format(list(labels.keys())[i]))
        plt.legend(loc = 'upper right')
    plt.suptitle('The mean value of different loss in erery epoch')
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    plt.savefig(osp.join(work_dir,f'visualize_loss_{timestamp}.svg'),format="svg")
    plt.cla()

