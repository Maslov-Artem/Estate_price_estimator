from airflow.decorators import dag, task
from datetime import datetime
from cian_parser import Parser
from airflow.providers.postgres.hooks.postgres import PostgresHook

default_args = {"owner": "Artem",
                "start_date": datetime(2024, 4, 4)}

@dag(dag_id="estate_data", default_args=default_args, schedule_interval="@daily",tags=['cian'])
def get_estate_data_task():

        
    @task(task_id = "parse_estate")
    def parse_and_insert():
        proxies = [
    {"http": "http://MTdPVT:1vaBRr@46.161.44.120:9252", "https": "http://MTdPVT:1vaBRr@46.161.44.120:9252"},
    {"http": "http://MTdPVT:1vaBRr@46.161.47.209:9262", "https": "http://MTdPVT:1vaBRr@46.161.47.209:9262"},
    {"http": "http://gJkm1m:Z4NDKQ@46.161.46.33:9003", "https": "http://gJkm1m:Z4NDKQ@46.161.46.33:9003"},
            {'http': 'http://sHmRxS:WojnT6@188.130.201.41:8000', 'https': 'http://sHmRxS:WojnT6@188.130.201.41:8000'}
]

        parser = Parser(proxies)
        
        hook = PostgresHook(postgres_conn_id="estate_data")

        for min_price, max_price in parser.small_apt_price_ranges.items():
            data = parser.parse(parser.small_apt_rooms, min_price, max_price)

            for estate in data:
                for key, value in estate.items():
                    if isinstance(value, str):
                        estate[key] = value.replace("'", "`")
                query = """
        INSERT INTO real_estate (offer_type, city, okrug, raion, street, house, room_count, room_type, loggiasCount,
                                  latitude, longitude, id, phone, flat_type, is_apartment, is_penthause, total_area,
                                  living_area, kitchen_area, all_rooms_area, combined_wcs_count, repair_type, floor,
                                  price, currency, material_type, floors_count, build_year, ceiling_height,
                                  house_material_type, edit_time, publication_date, house_material_bti,
                                  house_heating_supply, is_emergency, house_overlap_type, metro, metro_time, travel_type)
        VALUES ('{offer_type}', '{city}', '{okrug}', '{raion}', '{street}', '{house}', {room_count}, '{room_type}', {loggiasCount},
                {latitude}, {longitude}, {id}, '{phone}', '{flat_type}', {is_apartment}, {is_penthause}, {total_area},
                {living_area}, {kitchen_area}, '{all_rooms_area}', {combined_wcs_count}, '{repair_type}', {floor},
                {price}, '{currency}', '{material_type}', {floors_count}, {build_year}, '{ceiling_height}',
                '{house_material_type}', '{edit_time}', {publication_date}, '{house_material_bti}',
                '{house_heating_supply}', {is_emergency}, '{house_overlap_type}', '{metro}', {metro_time}, '{travel_type}');
        """.format(**estate)
                hook.run(sql=query)


        for min_price, max_price in list(parser.big_apt_price_ranges.items())[1:]:
            data = parser.parse(parser.big_apt_rooms, min_price, max_price)
            for estate in data:
                for key, value in estate.items():
                    if isinstance(value, str):
                        estate[key] = value.replace("'", "`")
                query = """
        INSERT INTO real_estate (offer_type, city, okrug, raion, street, house, room_count, room_type, loggiasCount,
                                  latitude, longitude, id, phone, flat_type, is_apartment, is_penthause, total_area,
                                  living_area, kitchen_area, all_rooms_area, combined_wcs_count, repair_type, floor,
                                  price, currency, material_type, floors_count, build_year, ceiling_height,
                                  house_material_type, edit_time, publication_date, house_material_bti,
                                  house_heating_supply, is_emergency, house_overlap_type, metro, metro_time, travel_type)
        VALUES ('{offer_type}', '{city}', '{okrug}', '{raion}', '{street}', '{house}', {room_count}, '{room_type}', {loggiasCount},
                {latitude}, {longitude}, {id}, '{phone}', '{flat_type}', {is_apartment}, {is_penthause}, {total_area},
                {living_area}, {kitchen_area}, '{all_rooms_area}', {combined_wcs_count}, '{repair_type}', {floor},
                {price}, '{currency}', '{material_type}', {floors_count}, {build_year}, '{ceiling_height}',
                '{house_material_type}', '{edit_time}', {publication_date}, '{house_material_bti}',
                '{house_heating_supply}', {is_emergency}, '{house_overlap_type}', '{metro}', {metro_time}, '{travel_type}');
        """.format(**estate)
                hook.run(sql=query)
 
    parse_and_insert()


get_estate_data_task()
