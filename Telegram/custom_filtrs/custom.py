from typing import Union

from aiogram.filters import BaseFilter
from aiogram.types import Message
import string

class ID_Filter(BaseFilter):

    def __init__(self):
        self.no = string.ascii_letters + string.punctuation + ' _-/.,\\'
        self.r = ''

    async def __call__(self, message: Message) -> bool:

        text = message.text
        print(text, type(text))
        self.r = ''
        for i in text:
            if i not in self.no:
                self.r += i
        
        if 7 < len(self.r) < 10:
            return str(int(self.r)) == self.r
        else:
            return False
        

class Square_F(BaseFilter):
    def __init__(self):
        self.no = string.ascii_letters + string.punctuation + ' _-/.,\\'
        self.r = ''
    async def __call__(self, message: Message) -> bool:
        text = message.text
        print(text, type(text))
        self.r = ''
        for i in text:
            if i not in self.no:
                self.r += i
        if 1 <= len(self.r)  <= 4:
            return str(int(self.r)) == self.r
        else:
            return False
        

class Square_F(BaseFilter):
    def __init__(self):
        self.no = string.ascii_letters + string.punctuation + ' _-/.,\\'
        self.r = ''
    async def __call__(self, message: Message) -> bool:
        text = message.text
        print(text, type(text))
        self.r = ''
        for i in text:
            if i not in self.no:
                self.r += i
        if 1 <= len(self.r)  <= 4:
            return str(int(self.r)) == self.r
        else:
            return False