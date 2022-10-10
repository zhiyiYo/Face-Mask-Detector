# coding:utf-8
import json
from pathlib import Path


class Config:
    """ 配置类 """

    folder = Path('app/config')

    def __init__(self):
        self.__config = {
            "model": "",
            "enable-acrylic": True,
            "enable-gpu": True,
            "confidence-threshold": 0.5,
        }
        self.__readConfig()

    def __readConfig(self):
        """ 读入配置文件数据 """
        try:
            with open("app/config/config.json", encoding="utf-8") as f:
                self.__config.update(json.load(f))
        except:
            pass

    def __setitem__(self, key, value):
        if key not in self.__config:
            raise KeyError(f'配置项 `{key}` 非法')

        if self.__config[key] == value:
            return

        self.__config[key] = value
        self.save()

    def __getitem__(self, key):
        return self.__config[key]

    def copy(self):
        """ 拷贝配置 """
        return self.__config.copy()

    def update(self, config: dict):
        """ 更新配置 """
        for k, v in config.items():
            self[k] = v

    def save(self):
        """ 保存配置 """
        self.folder.mkdir(parents=True, exist_ok=True)
        with open("app/config/config.json", "w", encoding="utf-8") as f:
            json.dump(self.__config, f)
