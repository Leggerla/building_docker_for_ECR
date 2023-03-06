# syntax=docker/dockerfile:1

FROM public.ecr.aws/lambda/python:3.8

ENV NUMBA_CACHE_DIR=/tmp/numba_cache

RUN  pip install --upgrade pip

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"


RUN mkdir .u2net && curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o .u2net/u2net.onnx
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]