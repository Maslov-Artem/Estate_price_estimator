import json
import requests


from aiogram import Router, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import StatesGroup, State
from aiogram.types import Message, ReplyKeyboardRemove
from custom_filtrs.custom import Square_F
from funcs.cian_m import gimi_smth
from funcs.geo_coder import coord_tg
from funcs.const import *

from keyboards.simple_row import make_row_keyboard




router = Router()


class CondState(StatesGroup):
    cond_1 = State()
    cond_2 = State()
    cond_3 = State()


@router.message(StateFilter(None), F.text == catch_1)
async def cian_id(message: Message, state: FSMContext):
    await message.answer(
        text="Укажите примерную площадь: ",
        reply_markup=None
    )
    await state.set_state(CondState.cond_1)



@router.message(CondState.cond_1, Square_F())
async def cond_answer(message: Message, state: FSMContext):

    await state.update_data(square=message.text)

    await message.answer(
        text="Оцените качество ремонта: ",
        reply_markup=make_row_keyboard(available_quality_names)
    )
    await state.set_state(CondState.cond_2)



@router.message(CondState.cond_2, F.text.in_(available_quality_names))
async def cond_answer_2(message: Message, state: FSMContext):

    await state.update_data(quality=message.text.lower())

    await message.answer(
        text="Кидай локацию: адрес или булавка 📍",
        reply_markup=None
    )
    await state.set_state(CondState.cond_3)

@router.message(CondState.cond_3, F.location)
async def handle_location(message: Message, state: FSMContext):
    lati = message.location.latitude
    loni = message.location.longitude

    await state.update_data(lat=lati)
    await state.update_data(lon=loni)

    user_data = await state.get_data()

    user_data = await state.get_data()
    
    forward = {
        'square': user_data['square'],
               'quality':user_data['quality'],
            #    'adress':user_data['adress'],
               'lat': user_data['lat'],
               'lan': user_data['lon'],
               }


    res = requests.post(url, json=forward)
    res = json.loads(res.content)['classify_me']

    reply = f'Справедливая стоимость данного объекта: {res}₽'
    await message.answer(reply)

    await message.answer('/start')



    await state.clear()



@router.message(CondState.cond_3)
async def cond_answer_3(message: Message, state: FSMContext):

    place = message.text

    if coord_tg(place) is None:
        reply = 'Введите адрес еще раз или нажмите /cancel'
        await message.answer(reply)

    lati, loni = coord_tg(place)

    


    await message.answer_location(latitude = lati , longitude = loni)

    await state.update_data(lat=lati)
    await state.update_data(lon=loni)
    await state.update_data(adress=message.text)

    user_data = await state.get_data()
    
    forward = {
        'square': user_data['square'],
               'quality':user_data['quality'],
            #    'adress':user_data['adress'],
               'lat': user_data['lat'],
               'lan': user_data['lon'],
               }


    res = requests.post(url, json=forward)
    res = json.loads(res.content)['classify_me']

    reply = f'Справедливая стоимость данного объекта: {res}₽'
    await message.answer(reply)

    await message.answer('/start')



    await state.clear()




################################################################
@router.message(CondState.cond_1,)
async def food_chosen(message: Message, state: FSMContext):

    await message.answer(f'Введите площадь ещё раз или нажмите \cancel')

@router.message(CondState.cond_2)
async def cond_answer_2(message: Message, state: FSMContext):

    await message.answer(
        text="Что-то пошло не так :(\nОцените качество ремонта или нажмите \cancel:",
        reply_markup=make_row_keyboard(available_quality_names)
    )