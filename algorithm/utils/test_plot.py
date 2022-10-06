# coding:utf-8
import matplotlib as mpl
import matplotlib.pyplot as plt

from utils.plot_utils import *

mpl.rc_file('resource/theme/matlab.mplstyle')

plot_mAP("eval/mAPs.json")
plt.show()
