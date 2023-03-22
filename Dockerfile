FROM openjdk:8-jre-slim

RUN apt-get update -y && \
	apt-get install -y python3 && \
	ln -s /usr/bin/python3 /usr/bin/python && \
	rm -rf /var/lib/apt/list/*

ARG jupyterlab_version=2.1.5

RUN apt-get update -y && \
	apt-get install -y python3-pip
	
RUN pip3 install wget jupyterlab==${jupyterlab_version}

RUN pip3 install tabulate numpy pandas scikit-learn matplotlib seaborn linearmodels lightgbm xgboost

RUN pip3 install graphviz pydot  statsmodels scipy lightgbm

RUN mkdir /fs
RUN mkdir /ssl
EXPOSE 8888
WORKDIR /fs

ARG HASH=argon2:$argon2id$v=19$m=10240,t=10,p=8$wbkIJV8mUeJRTQth/JZlgQ$EGbWw+NEu7qXSx6d94BeaH9O0U5YiMSHWT37QSLfesI

RUN echo ${HASH}

CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --certfile=/etc/letsencrypt/live/snf-34626.ok-kno.grnetcloud.net/fullchain.pem --keyfile=/etc/letsencrypt/live/snf-34626.ok-kno.grnetcloud.net/privkey.pem --NotebookApp.token=datasource2023

#CMD bash
