FROM nvidia/cuda:8.0-cudnn6-runtime

LABEL maintainer "takayama.junya@ist.osaka-u.ac.jp"

RUN apt-get update
RUN apt-get install -y locales

RUN apt-get install language-pack-ja-base language-pack-ja
RUN locale-gen ja_JP.UTF-8
RUN echo export LANG=ja_JP.UTF-8 >> ~/.profile
RUN . ~/.profile

RUN apt-get install -y python3-pip curl
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow-gpu==1.3.0
RUN pip3 install keras==2.1.5
RUN pip3 install folium

RUN pip3 install jupyter
RUN apt-get install -y aptitude
RUN pip3 install pandas matplotlib scikit-learn scipy
RUN apt-get install -y emacs24
RUN apt-get install -y git sudo

VOLUME /notebook
WORKDIR /notebook
EXPOSE 8888
CMD jupyter notebook --no-browser --ip=0.0.0.0 --allow-root