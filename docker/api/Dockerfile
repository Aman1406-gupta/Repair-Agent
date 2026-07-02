# FROM prod-nexus.sprinklr.com:8123/intuition/intuition_base_amd_any_py311:1.0 
FROM python:3.11

RUN mkdir -p /api/agent_builder
WORKDIR /api/agent_builder

RUN apt-get update

RUN apt-get install -y unbound=1.22.0-2+deb13u3

COPY ./docker/api/requirements.txt ./requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver -r requirements.txt

COPY ./ /api/agent_builder/

ENV PYTHONPATH=/api
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

CMD ["/bin/bash"]
