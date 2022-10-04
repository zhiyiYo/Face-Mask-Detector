# coding:utf-8
from datetime import datetime


def time_delta(t: datetime):
    """ 计算现在到指定时间的间隔 """
    dt = datetime.now()-t
    hours = dt.seconds//3600
    minutes = (dt.seconds-hours*3600) // 60
    seconds = dt.seconds % 60
    return f'{hours:02}:{minutes:02}:{seconds:02}'

