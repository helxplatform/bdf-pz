FROM python:3.12

USER root

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN echo "deb http://deb.debian.org/debian bullseye main" > /etc/apt/sources.list.d/bullseye.list

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get upgrade -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y -t bullseye libnss-ldap/bullseye && \
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
        gnupg \
        rsync \
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

# Install `node 20.x`.
# This is necessary to build beaker's UI bundle, which is not present nor
# automatically built if installing from the git repository directly.
RUN mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
  && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" > /etc/apt/sources.list.d/nodesource.list \
  && apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y nodejs \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir hatch pip
RUN pip install --no-cache-dir cython

# Lock palimpzest dependencies to specific versions. This greatly reduces the build latency introduced by pip's dependency
# resolution. Ideally, this should be abstracted into something like Poetry eventually to avoid hard-coded dependencies.
RUN pip install -v --no-cache-dir --no-deps "git+https://github.com/helxplatform/palimpzest.git@vllm-completion-bug"
# Use CPU-only vesrion of torch for sentence-transformers. Currently, we aren't using local inference (only text-embedding-3)
# so don't need to pull in a bunch of CUDA libs.
RUN pip install -v --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN pip install -v --no-cache-dir \
    "chromadb==1.2.1" \
    "colorama==0.4.6" \
    "fastapi==0.120.0" \
    "langchain==0.3.27" \
    "litellm==1.78.7" \
    "numpy==2.0.2" \
    "openai==2.6.0" \
    "pandas==2.3.3" \
    "pillow==11.3.0" \
    "prettytable==3.16.0" \
    "psutil==7.1.1" \
    "PyLD==2.0.4" \
    "pypdf==6.1.3" \
    "pyyaml==6.0.3" \
    "requests==2.32.5" \
    "sentence-transformers==5.0.0" \
    "smolagents[toolkit]==1.22.0" \
    "tqdm==4.67.1" \
    "rich[jupyter]==14.2.0" \
    "psycopg2" \
    "pydot==4.0.1" \
    "graphviz==0.21" \
    "pygraphviz==1.14" \
    "scipy==1.16.2" \
    "seaborn==0.13.2"

COPY root /

COPY . /jupyter/
# Install project dependencies prior to installing the project. Beaker-kernel's hatch builder
# will attempt to import project files, and this will throw if every dependency is not preinstalled.
RUN python /extract-deps.py /jupyter/pyproject.toml \
    # Ignore palimpzest, which we lock the dependencies of in a previous step.
    # Ignore beaker, since we install from a git repository, there is no UI bundle.
    # Needs to be editable to get build the UI bundle (otherwise build tools and UI source is omitted).
    | grep -viE '^(palimpzest|beaker)' > requirements.txt && \
    pip install -v --no-cache-dir -r requirements.txt && \
    # Beaker-kernel requires `zmq` as a dependency but does not specify it in its `project.dependencies` list.
    pip install -v --no-cache-dir zmq && \
    rm requirements.txt
RUN pip install -v -e "git+https://github.com/helxplatform/beaker-kernel.git@fix-fail-task#egg=beaker_kernel"
# All project dependencies are already installed by extract-deps, no need to reinstall the exact same list.
RUN pip install -v --no-cache-dir --no-build-isolation --no-deps /jupyter

RUN mkdir -m 777 /var/run/beaker

# Set default server env variables
ENV NB_UID=1000
ENV NB_GID=0
ENV BEAKER_RUN_PATH=/var/run/beaker
ENV BEAKER_APP=bdf_pz.app.PalimpzestApp

# Since beaker is being installed as a git package, it won't include a prebuilt UI bundle. This needs to be built manually.
RUN /build-ui-bundle.sh
# Need to fix beaker's site-package permissions so that its UI bundle can be patched at runtime by fix-ui-bundle.sh
RUN /fix-permissions.sh $(python -c "import beaker_kernel; import os; print(os.path.dirname(beaker_kernel.__file__))")
RUN /fix-permissions.sh "/usr/local/share/beaker"
RUN /fix-permissions.sh "/home"

USER helx

CMD ["/start.sh"]