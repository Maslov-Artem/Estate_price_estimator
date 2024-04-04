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
@router.message(Command("restart"))
@router.message(F.text.lower() == "start")
async def proccess_command_start(message: Message, state: FSMContext):
    await state.clear()
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
        text="Действие отменено\n/restart",
        reply_markup=ReplyKeyboardRemove()
    )


@router.message(F.text.lower() == "state")
async def cmd_state(message: Message, state: FSMContext):
    user_data = await state.get_data()
    await message.answer(
        text=f'date has been printed\n{user_data}\n/restart',
        reply_markup=ReplyKeyboardRemove()
    )
import requests
import json

@router.message(Command(commands=["retry"]))
@router.message(F.text.lower() == "retry")
async def cmd_retry(message: Message):

    data = {'square':50, 'quality':'no','lat':56.10,'lan':38.24}
    await message.answer('/retry\n/restart')


    res = requests.post(url, json=data)
    back = res.content

    # reply = f'Справедливая стоимость данного объекта: {res}₽'
    await message.answer(back)


@router.message(Command(commands=["help"]))
@router.message(F.text.lower() == "help")
async def cmd_help(message: Message):

    
    await message.answer('/cancel\n\n/restart\n\n/state\n\n/retry\n\n/help')