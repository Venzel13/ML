FROM jupyter/scipy-notebook:17aba6048f44
COPY . /home/jovyan
RUN pip install -r requirements.txt
EXPOSE 8080
VOLUME [ "./vol" ]