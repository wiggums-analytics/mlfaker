FROM manifoldai/orbyter-ml-dev:3.3
COPY requirements.txt /build/requirements.txt
WORKDIR /build/
RUN pip install -r requirements.txt
WORKDIR /mnt/
COPY setup.py tox.ini myproject.toml /mnt/
COPY ./mlfaker /mnt/mlfaker
RUN pip install -e /mnt/
