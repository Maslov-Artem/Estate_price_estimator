from aiogram import Router, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, ReplyKeyboardRemove
import requests
from custom_filtrs.custom import ID_Filter
from funcs.cian_m import gimi_smth
from funcs.const import *

router = Router()


class CianState(StatesGroup):
    cian_id_true = State()

@router.message(StateFilter(None), F.text == catch_2)
async def cian_id(message: Message, state: FSMContext):
    await message.answer(
        text="Укажите ID:",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(CianState.cian_id_true)


@router.message(CianState.cian_id_true, ID_Filter())
async def cian_answer(message: Message, state: FSMContext):

    r = ''
    txt = str(message.text)
    for char in txt:
        if char.isdigit():
            r += char

    json = {'find_me':r}



    res = requests.post(url_cian, json=json)
    json = res.json()

    if str(json['pred']) == '-1.0':
        reply = f'Данной ID нет в базе: отправьте ещё один или нажмите /start'
    else:
        pred = int(float(json["pred"]))
        price = int(float(json["real"]))
        if price > 1.15*pred:
            
            dop = f'Кажется, цена в объявлении сильно завышена'
        elif price*1.05 < pred:
            dop = f'Кажется, цена в объявлении нерыночная. Будьте бдительны.'
        else:
            dop = f'Цена в объявлении вполне себе рыночная.'



        reply = f'''Цена в объявлении: {price} ₽.
Справедливая стоимость данного объекта: {int(float(json["pred"]))} ₽.
Объект находится в {round(float(json["metro_dist"]),2)} км у станции {json["metro"].title()}
До центра Москвы {round(float(json['kremlin_dist']),2)} км
{dop}

Отправьте ещё один или нажмите /start
Ссылка: {json['link']}'''
        
    await message.answer(reply)
    lan, lat = float(json['lan']), float(json['lat'])
    if lan != -1:
        await message.answer_location(latitude = lat , longitude = lan)



################################################################
@router.message(CianState.cian_id_true)
async def food_chosen(message: Message):

    await message.answer(f'''{message.text} кажется id нанастоящий...\nКинь ещё один или начни сначала: /start''')