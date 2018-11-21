import numpy as np
import matplotlib.pyplot as plt
from baselines.results_plotter import load_results, ts2xy
import argparse
import os
import ipdb


def movingAverage(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    # print('log_folder: ',log_folder)
    res=load_results(log_folder)
    # timesteps ts2xy not working
    x=np.cumsum(res.l.values)
    y=res.r.values
    y = movingAverage(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()



def plot_multi(log_folder, log_names, title='Reinforcement Curve'):

    xs=[]
    ys=[]
    # ipdb.set_trace()
    for log_name in log_names:
        res=os.path.join(log_folder, log_name)

        df=load_results(res)
        x=np.cumsum(df.l.values)
        y=df.r.values
        y=movingAverage(y, window=50)
        x=x[len(x)-len(y):]

        xs.append(x)
        ys.append(y)

    # ipdb.set_trace()
    minlen=np.inf
    for item in xs:
        length=item.shape[0]
        if length<minlen:
            minlen=length

    for i in range(len(xs)):
        xs[i]=xs[i][0:minlen]
        ys[i]=ys[i][0:minlen]


    meanx=np.mean(xs, axis=0)
    meany=np.mean(ys, axis=0)
    stdx=np.std(xs, axis=0)
    stdy=np.std(ys,axis=0)


    fig=plt.figure(title)
    plt.grid()
    plt.plot(meanx, meany, 'o-', color='r', label='mean training curves', markersize=3)
    plt.fill_between(meanx, meany-stdy, meany+stdy, alpha=0.1, color='r')
    plt.legend(loc='best')
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title+"Smoothed")
    plt.show()




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base', default='/tmp/gym/', help='base dir')
    parser.add_argument('--log', default='test/', help='log folder')
    args=parser.parse_args()

    # filename=os.path.join(args.base, args.log)
    # print('print log results...')
    # plot_results(filename)

    log_folder=args.base
    # log_names=['test1/','test/']
    # log_names=['test/','test1/','test2/']
    log_names=['rarl_2/','rarl_1/']
    print('plot log resutls...')
    plot_multi(log_folder, log_names)