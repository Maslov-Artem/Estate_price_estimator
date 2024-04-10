from aiogram import Router, F, types
from aiogram.utils.keyboard import InlineKeyboardBuilder

from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, ReplyKeyboardRemove
import requests
from custom_filtrs.custom import ID_Filter
from funcs.cian_m import gimi_smth
from funcs.const import *

url_c = url+'cian_id'

router = Router()


class CianState(StatesGroup):
    cian_id_true = State()
    cian_2 = State()

@router.message(StateFilter(None), F.text == catch_2)
async def cian_id(message: Message, state: FSMContext):
    await message.answer(
        text="Укажите ID:",
        reply_markup=ReplyKeyboardRemove()
    )
    await state.set_state(CianState.cian_id_true)


@router.message(CianState.cian_id_true or CianState.cian_2, ID_Filter())
async def cian_answer(message: Message, state: FSMContext):

    LNK = False
    r = ''
    txt = str(message.text)
    for char in txt:
        if char.isdigit():
            r += char
        else:
            LNK = True

    await state.update_data(link=r)

    data = {'find_me':r}

    res = requests.post(url_c, json=data)
    json = res.json()


    if str(json['pred']) == '-1.0':
        reply = f'Данной ID нет в базе: отправьте ещё один или нажмите /restart'
        await message.answer(text=reply)

        await state.set_state(CianState.cian_id_true)


        
    else:
        lan, lat = float(json['lan']), float(json['lat'])

        await state.update_data(lan=lan)
        await state.update_data(lat=lat)

        pred = int(float(json["pred"]))
        price = int(float(json["real"]))


        if price > 1.15*pred:
            dop = f'Кажется, цена в объявлении сильно завышена'
        elif price*1.05 < pred:
            dop = f'Кажется, цена в объявлении нерыночная. Будьте бдительны.'
        else:
            dop = f'Цена в объявлении вполне себе рыночная.'


        metro = json["metro"].split()[1]
        
        reply = f'''Цена в объявлении: {price} ₽.
Справедливая стоимость данного объекта: {int(float(json["pred"]))} ₽.

Объект находится в {round(float(json["metro_dist"]),2)} км у станции {metro.title()}

До центра Москвы {round(float(json['kremlin_dist']),2)} км

{dop}

Отправьте ещё один или нажмите /restart
'''
        
        builder = InlineKeyboardBuilder()
        builder.add(types.InlineKeyboardButton(
            text="✅",
            callback_data="location")
        )

        await message.answer(text=reply)
        await message.answer(text='Запрос локации', reply_markup=builder.as_markup())


        builder2 = InlineKeyboardBuilder()
        builder2.add(types.InlineKeyboardButton(
            text="🌐",
            callback_data="link")
        )

        if not LNK:
            await message.answer(
            "Ссылка?",
            reply_markup=builder2.as_markup())


        await state.set_state(CianState.cian_2)





@router.callback_query(F.data == "link", CianState.cian_2)
async def send_location(callback: types.CallbackQuery, state: FSMContext):

    data = await state.get_data()

    link = f'https://www.cian.ru/sale/flat/{data["link"]}/'

    await callback.message.answer(text=link)
    await callback.message.delete()

    await state.update_data(linki=1)

    try:
        if data['loci'] == 1:
            await state.clear()
            await state.set_state(CianState.cian_id_true)

    except:
        pass

@router.callback_query(F.data == "location", CianState.cian_2)
async def send_location(callback: types.CallbackQuery, state: FSMContext):

    data = await state.get_data()

    await callback.message.answer_location(latitude=data['lat'], longitude=data['lan'])
    await callback.message.delete()

    await state.update_data(loci=1)

    try:
        if data['linki'] == 1:
            await state.clear()
            await state.set_state(CianState.cian_id_true)

    except:
        pass


    


################################################################
@router.message(CianState.cian_id_true)
async def food_chosen(message: Message):

    await message.answer(f'''{message.text} кажется id нанастоящий...\nКинь ещё один или начни сначала: /restart''')