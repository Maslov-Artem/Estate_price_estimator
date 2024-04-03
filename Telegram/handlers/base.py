from aiogram import types
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.filters.command import Command
from aiogram import Router, F
from config_reader import config
from aiogram.utils.keyboard import InlineKeyboardBuilder
from funcs.const import *

import logging



router = Router()

@router.message(Command("start"))
async def proccess_command_start(message: Message):
    user_name = message.from_user.full_name
    user_id = message.from_user.id

    text = f'''Привет, {user_name}!
Это бот предназначен для оценки стоимости объекта недвижимости в Москве.
Для продолжения работы необходимо ответить на пару вопросов.
Также бот может оценить объявление на ЦИАНе. Для этого нужен всего лишь ID объявления.'''
    
    logging.info(f'{user_name} {user_id} запустил бота')
    kb = [
        [KeyboardButton(text=catch_1)],
        [KeyboardButton(text=catch_2)]
    ]

    keyboard = ReplyKeyboardMarkup(keyboard=kb, resize_keyboard=True,input_field_placeholder="...")

    await message.answer(text=text)
    await message.answer("Что выбираем?", reply_markup=keyboard)


@router.message(Command(commands=["cancel"]))
@router.message(F.text.lower() == "отмена")
async def cmd_cancel(message: Message, state: FSMContext):
    await state.clear()
    await message.answer(
        text="Действие отменено\n/start",
        reply_markup=ReplyKeyboardRemove()
    )


@router.message(F.text.lower() == "state")
async def cmd_cancel(message: Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer(
        text='date has been printed',
        reply_markup=ReplyKeyboardRemove()
    )
import requests
import json


@router.message(F.text.lower() == "retry")
async def cmd_cancel(message: Message, state: FSMContext):

    data = {'square':'50', 'quality':'Евроремонт','lat':'56.10','lan':'38.24'}


    res = requests.post(url, json=data)
    res = json.loads(res.content)['classify_me']

    reply = f'Справедливая стоимость данного объекта: {res}₽'
    await message.answer(reply)