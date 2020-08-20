FROM python:3.7.3-slim as covid19
WORKDIR COVID-19/
COPY requirements.txt .   # or "/" instead of "." ?
RUN pip install -r requirements.txt
COPY . . # either "/" or "." or "./" ?
# EXPOSE # указать порт в Dash и тут , чтобы запускалось автоматически
EXPOSE 8050:8050
VOLUME [ "." ]
ENTRYPOINT [ "python" ]
CMD [ "dashboard.py" ]