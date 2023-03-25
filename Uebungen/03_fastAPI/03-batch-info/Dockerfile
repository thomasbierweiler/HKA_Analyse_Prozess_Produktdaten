# 
FROM python:3.9
# 
WORKDIR /code
# 
COPY ./requirements.txt /code/requirements.txt
#
ENV PYTHONDONTWRITEBYTECODE=1
# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt --no-compile --no-cache-dir
# 
COPY ./app /code/app
#
WORKDIR /code/app
#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
