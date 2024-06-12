FROM rayproject/ray-ml:2.24.0-py310-gpu

RUN sudo apt-get update && sudo apt-get install -y nano

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY app.py /app/app.py