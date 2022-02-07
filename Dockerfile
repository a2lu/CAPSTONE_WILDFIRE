# 1) choose base container
# generally use the most recent tag

# base notebook, contains Jupyter and relevant tools
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2021.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt-get -y install aria2 nmap traceroute

# 3) install packages using notebook user
USER jovyan

RUN pip install --no-cache-dir geopandas
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir copy
RUN pip install --no-cache-dir datetime
RUN pip install --no-cache-dir scikit-learn