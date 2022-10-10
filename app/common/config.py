# coding: utf-8
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Union

from .exception_handler import exceptionHandler
from .singleton import Singleton


class ConfigValidator:
    """ Config validator """

    def validate(self, value) -> bool:
        """ Verify whether the value is legal """
        return True

    def correct(self, value):
        """ correct illegal value """
        return value


class RangeValidator(ConfigValidator):
    """ Range validator """

    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.range = (min, max)

    def validate(self, value) -> bool:
        return self.min <= value <= self.max

    def correct(self, value):
        return min(max(self.min, value), self.max)


class OptionsValidator(ConfigValidator):
    """ Options validator """

    def __init__(self, options: Union[Iterable, Enum]) -> None:
        if not options:
            raise ValueError("The `options` can't be empty.")

        if isinstance(options, Enum):
            options = options._member_map_.values()

        self.options = list(options)

    def validate(self, value) -> bool:
        return value in self.options

    def correct(self, value):
        return value if self.validate(value) else self.options[0]


class BoolValidator(OptionsValidator):
    """ Boolean validator """

    def __init__(self):
        super().__init__([True, False])


class FileValidator(ConfigValidator):
    """ File path validator """

    def validate(self, value: str) -> bool:
        return Path(value).exists()

    def correct(self, value: str):
        path = Path(value)
        if not path.exists() or path.is_dir():
            return ""

        return str(path.absolute()).replace("\\", "/")


class FolderValidator(ConfigValidator):
    """ Folder validator """

    def validate(self, value: str) -> bool:
        return Path(value).exists()

    def correct(self, value: str):
        path = Path(value)
        path.mkdir(exist_ok=True, parents=True)
        return str(path.absolute()).replace("\\", "/")


class FolderListValidator(ConfigValidator):
    """ Folder list validator """

    def validate(self, value: List[str]) -> bool:
        return all(Path(i).exists() for i in value)

    def correct(self, value: List[str]):
        folders = []
        for folder in value:
            path = Path(folder)
            if path.exists():
                folders.append(str(path.absolute()).replace("\\", "/"))

        return value


class ColorValidator(ConfigValidator):
    """ RGB color validator """

    def __init__(self, default: List[int]):
        self.default = default

    def validate(self, value: List[int]) -> bool:
        if not isinstance(value, list) or len(value) != 3:
            return False

        return all(0 <= i <= 255 for i in value)

    def correct(self, value: List[int]):
        return value if self.validate(value) else self.default


class ConfigSerializer:
    """ Config serializer """

    def serialize(self, value):
        """ serialize config value """
        return value

    def deserialize(self, value):
        """ deserialize config from config file's value """
        return value


class EnumSerializer(ConfigSerializer):
    """ enumeration class serializer """

    def __init__(self, enumClass):
        self.enumClass = enumClass

    def serialize(self, value: Enum):
        return value.value

    def deserialize(self, value):
        return self.enumClass(value)


class ConfigItem:
    """ Config item """

    def __init__(self, group: str, name: str, default, validator: ConfigValidator = None,
                 serializer: ConfigSerializer = None):
        """
        Parameters
        ----------
        group: str
            config group name

        name: str
            config item name, can be empty

        default:
            default value

        options: list
            options value

        serializer: ConfigSerializer
            config serializer
        """
        self.group = group
        self.name = name
        self.validator = validator or ConfigValidator()
        self.serializer = serializer or ConfigSerializer()
        self.__value = default
        self.value = default

    @property
    def value(self):
        """ get the value of config item """
        return self.__value

    @value.setter
    def value(self, v):
        self.__value = self.validator.correct(v)

    @property
    def options(self):
        """ get optional values, only available for item with `OptionsValidator` """
        if isinstance(self.validator, OptionsValidator):
            return self.validator.options

        return []

    @property
    def range(self):
        """ get the available range of config """
        if isinstance(self.validator, RangeValidator):
            return self.validator.range

        return (self.value, self.value)

    @property
    def key(self):
        """ get the config key separated by `.` """
        return self.group+"."+self.name if self.name else self.group

    def serialize(self):
        return self.serializer.serialize(self.value)

    def deserializeFrom(self, value):
        self.value = self.serializer.deserialize(value)


class Config(Singleton):
    """ Config of app """

    folder = Path('app/config')
    file = folder/"config.json"

    # model
    modelPath = ConfigItem("Model", "Path", "", FileValidator())
    useGPU = ConfigItem("Model", "UseGPU", True, BoolValidator())
    confidenceThreshold = ConfigItem(
        "Model", "ConfidenceThreshold", 0.5, RangeValidator(0.01, 0.99))

    # main window
    enableAcrylicBackground = ConfigItem(
        "MainWindow", "EnableAcrylicBackground", False, BoolValidator())

    def __init__(self):
        self.load()

    @classmethod
    def get(cls, item: ConfigItem):
        return item.value

    @classmethod
    def set(cls, item: ConfigItem, value):
        if item.value == value:
            return

        item.value = value
        cls.save()

    @classmethod
    def toDict(cls, serialize=True):
        """ convert config items to `dict` """
        items = {}
        for name in dir(cls):
            item = getattr(cls, name)
            if not isinstance(item, ConfigItem):
                continue

            value = item.serialize() if serialize else item.value
            if not items.get(item.group):
                if not item.name:
                    items[item.group] = value
                else:
                    items[item.group] = {}

            if item.name:
                items[item.group][item.name] = value

        return items

    @classmethod
    def save(cls):
        cls.folder.mkdir(parents=True, exist_ok=True)
        with open(cls.file, "w", encoding="utf-8") as f:
            json.dump(cls.toDict(), f, ensure_ascii=False, indent=4)

    @exceptionHandler("config")
    def load(self):
        """ load config """
        try:
            with open(self.file, encoding="utf-8") as f:
                cfg = json.load(f)
        except:
            cfg = {}

        # map config items'key to item
        items = {}
        for name in dir(Config):
            item = getattr(Config, name)
            if isinstance(item, ConfigItem):
                items[item.key] = item

        # update the value of config item
        for k, v in cfg.items():
            if not isinstance(v, dict) and items.get(k) is not None:
                items[k].deserializeFrom(v)
            elif isinstance(v, dict):
                for key, value in v.items():
                    key = k + "." + key
                    if items.get(key) is not None:
                        items[key].deserializeFrom(value)

        if sys.platform != "win32":
            self.enableAcrylicBackground.value = False


config = Config()
