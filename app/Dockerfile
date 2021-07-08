ARG BASE_IMAGE_ARG
FROM ${BASE_IMAGE_ARG}

ARG USER=brandon

ENV DEBIAN_FRONTEND=noninteractive
ENV USER=${USER}
ENV HOME=/home/${USER}

WORKDIR ${HOME}

RUN useradd -M -s /bin/bash -N -u 1000 ${USER} \
    && chown -R ${USER}:users /usr/local/bin \
    && chown -R ${USER}:users ${HOME}

COPY clean-layer.sh /usr/bin/clean-layer.sh

RUN chmod a+rwx /usr/bin/clean-layer.sh

# node.js
RUN curl -sL "https://deb.nodesource.com/gpgkey/nodesource.gpg.key" | apt-key add - \
    && echo "deb https://deb.nodesource.com/node_14.x focal main" > /etc/apt/sources.list.d/nodesource.list \
    && apt-get -yq update \
    && apt-get -yq install --no-install-recommends \
    nodejs \
    && clean-layer.sh

RUN pip install --no-cache-dir \
    adanet \
    jupyterlab \
    matplotlib \
    pandas \
    seaborn \
    sklearn

# CMD ["sh","-c", "jupyter lab --notebook_dir=/app --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password=''"]
ENTRYPOINT ["python3", "basic.py"]