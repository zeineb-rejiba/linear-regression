FROM tensorflow/tensorflow
RUN pip install sklearn pandas matplotlib
ADD *.py /lin_reg/
WORKDIR /lin_reg
COPY data ./data
COPY figs ./figs
CMD ["python", "lin_reg.py"]