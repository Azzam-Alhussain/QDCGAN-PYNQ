FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ARG GID
ARG GNAME
ARG UNAME
ARG UID
ARG PASSWD

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get -y install curl tmux htop vim sudo git rsync nano lsb-release \ 
    wget build-essential g++-multilib


RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN apt-get install -y libxrender-dev
RUN apt-get install -y zsh
RUN apt-get install -y sshpass
RUN apt-get install -y unzip
RUN apt-get install -y zip
RUN apt-get -y -f install
RUN apt-get autoclean && apt-get autoremove && rm -rf /var/lib/apt/lists/*
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config


# Create the user account
RUN mkdir /home/$UNAME
RUN groupadd -g $GID $GNAME
RUN useradd -M -u $UID $UNAME -g $GNAME
RUN usermod -aG sudo $UNAME
RUN echo "$UNAME:$PASSWD" | chpasswd
RUN echo "root:$PASSWD" | chpasswd
RUN chown -R $UNAME:$GNAME /home/$UNAME

WORKDIR /home/$UNAME

# password less sudo
RUN echo "%sudo ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN pip install future==0.18.2
RUN pip install numpy==1.18.0
RUN pip install scipy==1.5.2
RUN pip install ipykernel==5.5.5
RUN pip install jupyter==1.0.0
RUN pip install matplotlib==3.3.1 --ignore-installed
RUN pip install pandas==1.1.5
RUN pip install scikit-learn==0.24.1
RUN pip install tqdm==4.61.1
RUN pip install dill==0.3.3
RUN pip install versioned-hdf5
RUN pip install brevitas


USER $UNAME

CMD ["bash"]


