FROM python:3.13

USER root

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        lsof \
        libpq-dev \
        curl \
        git \
        nano-tiny \
        vim-tiny \
        tzdata \
        unzip \
        graphviz-dev \
        # git-over-ssh
        openssh-client \
        # `less` is needed to run help in R
        # see: https://github.com/jupyter/docker-stacks/issues/1588
        less \
        # `nbconvert` dependencies
        # https://nbconvert.readthedocs.io/en/latest/install.html#installing-tex
        texlive-xetex \
        texlive-fonts-recommended \
        texlive-plain-generic \
        # Enable clipboard on Linux host systems
        xclip \
        # - Add necessary fonts for matplotlib/seaborn
        #   See https://github.com/jupyter/docker-stacks/pull/380 for details
        fonts-liberation \
        # - `pandoc` is used to convert notebooks to html files
        #   it's not present in the aarch64 Ubuntu image, so we install it here
        pandoc \
        # - bzip2 is necessary to extract the micromamba executable.
        bzip2 \
        ca-certificates \
        locales \
        # - `netbase` provides /etc/{protocols,rpc,services}, part of POSIX
        #   and required by various C functions like getservbyname and getprotobyname
        #   https://github.com/jupyter/docker-stacks/pull/2129
        netbase \
        wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    echo "C.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen

RUN pip install --upgrade --no-cache-dir hatch pip
RUN pip install beaker-kernel==1.12.0
RUN pip install cython
COPY . /jupyter/
RUN pip install /jupyter

RUN pip install \
    "psycopg2" \
    "pydot==4.0.1" \
    "graphviz==0.21" \
    "pygraphviz==1.14" \
    "pandas>=2.1.2,<3.0.0" \
    "scipy>=1.11.3,<2.0.0" \
    "numpy==1.26.4"

RUN mkdir -m 777 /var/run/beaker

# Set default server env variables
ENV BEAKER_RUN_PATH=/var/run/beaker
ENV BEAKER_APP=bdf_pz.app.PalimpzestApp

COPY root /

USER helx

CMD ["/start.sh"]