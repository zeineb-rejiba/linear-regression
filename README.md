# Description

This repository contains the code for a Docker application that allows you to train a linear regression model using batch gradient descent. Both univariate and multivariate regression types are supported. The implementation is based on [Tensorflow](www.tensorflow.org) and [scikit-learn](https://scikit-learn.org/stable/). More specifically, Tensorflow is used for tensor operations such as multiplication and gradient calculations, whereas scikit-learn is used to provide feature scaling in the multivariate regression case. 

## Prerequisites

You only need [Docker](https://www.docker.com/) to run this application. You may find install instructions [here](https://docs.docker.com/desktop/), where you can choose the options corresponding to your operating system. 

## Usage
First, clone the repository.
```
git clone https://github.com/zeineb-rejiba/linear-regression.git
```
Then, go into the cloned project (containing the Dockerfile) and run the following command to build the image with a tag of your choice (here it is `regression-im` ):

    docker build -t regression-im .

 - **N.B.**: If you are using Windows, the following step needs to be done to allow for data sharing between the container and the host using volumes (e.g. to be able to access the figures in the `figs` directory.)
In Docker Desktop, go to `Settings>Resources>FILE SHARING` and specify the project's directory (i.e. `your_download_path\linear-regression`).


Then, you can run the container using the following command:

    docker run -v %cd%\figs:/lin_reg/figs --name regression regression-im

Note that `%cd%` in the previous commands corresponds to using Windows CLI. It should be replaced with:

 - `${PWD}` , if you are using Powershell.
 - `$(pwd)`, if you are using Linux.