FROM manifoldai/orbyter-ml-dev:3.2
COPY requirements.txt /build/requirements.txt
WORKDIR /build/
RUN pip install -r requirements.txt
WORKDIR /mnt/
COPY setup.py tox.ini myproject.toml /mnt/
COPY ../mlfaker /mnt/
RUN pip install -e /mnt/
