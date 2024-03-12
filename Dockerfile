FROM openjdk:8-jre-slim

ARG HADOOP_VERSION=2.9.1
ARG HADOOP_PREFIX=/usr/local/hadoop
ARG HADOOP_URL=https://archive.apache.org/dist/hadoop/core/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz



RUN \
    # Install dependencies.
    apt-get update -y && \
    apt-get install --yes --no-install-recommends procps curl gpg host netcat software-properties-common gnupg2 -y && \
    # Clean.
    apt-get clean autoclean && \
    apt-get autoremove --yes

RUN \
    # Create hadoop directory.
    mkdir -p ${HADOOP_PREFIX} && \
    # Download hadoop gpg keys.
    curl https://dist.apache.org/repos/dist/release/hadoop/common/KEYS -o HADOOP_KEYS && \
    gpg --import HADOOP_KEYS && \
    # Download, install hadoop.
    curl -fSL "${HADOOP_URL}" -o /tmp/hadoop.tar.gz && \
    curl -fSL "${HADOOP_URL}.asc" -o /tmp/hadoop.tar.gz.asc && \
    gpg --verify /tmp/hadoop.tar.gz.asc && \
    tar -C "${HADOOP_PREFIX}" --strip=1 -xzf /tmp/hadoop.tar.gz && \
    rm /tmp/hadoop.tar.gz*


# Set hadoop environment variables.
ENV HADOOP_HOME         "${HADOOP_PREFIX}"
ENV HADOOP_COMMON_HOME  "${HADOOP_PREFIX}"
ENV HADOOP_CONF_DIR     "${HADOOP_PREFIX}/etc/hadoop"
ENV HADOOP_HDFS_HOME    "${HADOOP_PREFIX}"
ENV HADOOP_MAPRED_HOME  "${HADOOP_PREFIX}"
ENV HADOOP_NAMENODE_DIR "/var/local/hadoop/hdfs/namenode/"
ENV HADOOP_DATANODE_DIR "/var/local/hadoop/hdfs/datanode/"

# Set YARN environment variables.
ENV HADOOP_YARN_HOME   "${HADOOP_PREFIX}"
ENV YARN_CONF_DIR      "${HADOOP_PREFIX}/etc/hadoop"


#RUN add-apt-repository ppa:deadsnakes/ppa
RUN add-apt-repository "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main"
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776

RUN apt-get update -y
RUN apt-get install -y python3.7 
RUN apt-get install -y python3.7-distutils

RUN ln -sf /usr/bin/python3.7 /usr/bin/python
RUN ln -sf /usr/bin/python3.7 /usr/bin/python3
#RUN rm -rf /var/lib/apt/list/*


RUN python3 --version

RUN apt-get install -y wget

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py

ARG jupyterlab_version=2.1.5
ARG spark_version=2.3.2

RUN python3 -m pip install --upgrade pip 	
RUN python3 -m pip install wget pyspark==${spark_version} jupyterlab==${jupyterlab_version}

RUN python3 -m pip install tabulate numpy pandas scikit-learn matplotlib seaborn linearmodels lightgbm xgboost

RUN python3 -m pip install graphviz pydot  statsmodels scipy lightgbm

RUN mkdir /fs
RUN mkdir /ssl
EXPOSE 8888
WORKDIR /fs

ARG HASH=argon2:$argon2id$v=19$m=10240,t=10,p=8$wbkIJV8mUeJRTQth/JZlgQ$EGbWw+NEu7qXSx6d94BeaH9O0U5YiMSHWT37QSLfesI

RUN echo ${HASH}

ADD ./hdfs-site.xml ./core-site.xml "${HADOOP_PREFIX}/etc/hadoop/"

ADD ./yarn-site.xml "${HADOOP_PREFIX}/etc/hadoop/"

# Set YARN environment variables.
ENV HADOOP_YARN_HOME   "${HADOOP_PREFIX}"
ENV YARN_CONF_DIR      "${HADOOP_PREFIX}/etc/hadoop"

RUN apt-get install -y libkrb5-dev 
RUn apt-get install -y heimdal-dev 
RUN apt-get install -y gcc 
RUN apt-get install -y python3.7-dev
RUN python3 -m pip install requests-kerberos
RUN python3 -m pip install sparkmagic


RUN mkdir /root/.sparkmagic
ADD ./config.json "/root/.sparkmagic"
RUN sed -i 's/localhost/spark/g' /root/.sparkmagic/config.json
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension
RUN jupyter-kernelspec install $(pip show sparkmagic | grep Location | cut -d" " -f2)/sparkmagic/kernels/sparkkernel
RUN jupyter-kernelspec install $(pip show sparkmagic | grep Location | cut -d" " -f2)/sparkmagic/kernels/pysparkkernel
RUN jupyter-kernelspec install $(pip show sparkmagic | grep Location | cut -d" " -f2)/sparkmagic/kernels/sparkrkernel
RUN jupyter serverextension enable --py sparkmagic



#CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --certfile=/etc/letsencrypt/live/snf-34626.ok-kno.grnetcloud.net/fullchain.pem --keyfile=/etc/letsencrypt/live/snf-34626.ok-kno.grnetcloud.net/privkey.pem --NotebookApp.token=datasource2023
CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=datasource2023

#CMD bash
