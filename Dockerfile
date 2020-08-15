# Use tensorflow as a base image
FROM tensorflow/tensorflow
# Install libraries
RUN pip install sklearn pandas matplotlib
# Add python files to lin_reg directory
ADD *.py /lin_reg/
# Make lin_reg the working directory
WORKDIR /lin_reg
# Copy data into the data directory in the container
COPY data ./data
# Copy figs into the figs directory in the container
COPY figs ./figs
# Define python lin_reg.py as the default command for the container
CMD ["python", "lin_reg.py"]