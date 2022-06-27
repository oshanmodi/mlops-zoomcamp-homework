FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

ENTRYPOINT [ "bash" ]
RUN pip install -U pip
RUN pip install pipenv

COPY ["Pipfile","Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

RUN mkdir tmp

RUN python starter.py 2021 4