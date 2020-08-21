
FROM python:3.7.3
#-slim as covid19
WORKDIR /COVID
# or "/" instead of "." ?
COPY requirements.txt .
RUN pip install -r requirements.txt
# either "/" or "." or "./" ?
COPY . .
# EXPOSE # указать порт в Dash и тут , чтобы запускалось автоматически
EXPOSE 8050:8050
# VOLUME [ "." ]
# ENTRYPOINT [ "python", "dashboard.py" ]