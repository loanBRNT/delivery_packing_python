FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN pip install uvicorn fastapi json-numpy

COPY . /

RUN pip install /pointnet2_ops/

RUN pip install -r /requirements.txt

RUN pip install /

RUN wget https://huggingface.co/wentao-yuan/m2t2/resolve/main/m2t2.pth -O /m2t2.pth

CMD ["python", "/home/run.py"]