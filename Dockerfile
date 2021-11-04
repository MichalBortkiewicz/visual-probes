FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG USER_NAME
ARG PASSWD
ARG UID
ARG PROJECT_NAME=visual-probes

RUN groupadd -g $UID $USER_NAME
RUN useradd -rm -d /home/$USER_NAME -s /bin/bash -g $UID -G sudo -u $UID $USER_NAME


# This prevents installers from opening dialog boxes, be careful this setting is inherited!
#ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get -y install \
    sudo \
    openssh-server \
    build-essential \
    cmake \
    ffmpeg \
    net-tools \
    pkg-config \
    protobuf-compiler \
    python3 \
    python3-pip \
    tmux \
    wget \
    nano \
    git \
    unzip


RUN mkdir /var/run/sshd
RUN echo 'root:'$PASSWD | chpasswd
RUN echo $USER_NAME:$PASSWD | chpasswd
#RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

#COPY authorized_keys /root/.ssh/authorized_keys

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]

WORKDIR /home/$USER_NAME/$PROJECT_NAME/

RUN sudo chown -R $USER_NAME /home/$USER_NAME

COPY . .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

