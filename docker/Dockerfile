FROM alpine as data_download
#Download paper dataset
RUN apk add wget && \
    mkdir -p /datasets/

ADD ./imagesimdl.sh /datasets/imagesimdl.sh
RUN mkdir -p /datasets/imagesim/images && \
    cd /datasets/imagesim && \
    sh /datasets/imagesimdl.sh && \
    cd /datasets && \
    echo "Downloaded datasets, size " $(du -sh ./imagesim)

#Download Tiny Imagenet
RUN apk add --no-cache unzip && \
    cd /datasets && \
    wget --progress=bar:force:noscroll -O tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip && \
    unzip tiny-imagenet-200.zip && \
    rm -rf tiny-imagenet-200.zip && \
    echo "Downloaded Tiny Imagenet"


FROM luntlab/bw-pytorch:v1.5.0-libs as project_base
MAINTAINER "Bryan J. Lunt <bjlunt2@illinois.edu>"

RUN apt update && \
    apt install -y git && \
    rm -rf /var/lib/apt/lists/*

RUN conda install -y scipy && \
    conda install -c conda-forge yaml && \
    conda install -y -c conda-forge wandb && \
    conda install -y scikit-learn jedi==0.17.2 pandas
    
RUN cd /tmp && \
    git clone https://github.com/PyTorchLightning/pytorch-lightning.git && \
    cd pytorch-lightning && \
    git checkout 44d775fccfb825561937f6fa03fe258af25c2b83 && \
    python setup.py install && \
    cd .. && \
    rm -rf pytorch-lightning

WORKDIR /workspace
COPY --from=data_download /datasets /workspace/datasets
