FROM python:3.11

WORKDIR /code

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8500

ENTRYPOINT [ "python", "bot.py" ]
