FROM python:3.11.6
ENV TOKEN='token from @BotFather'
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "test.py" ]