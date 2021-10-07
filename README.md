# MNIST_FPGA

Joash Naidoo

A tutorial project to try implement a very basic neural network which classifies the MNIST data set on a Kintex7 FPGA. The project relies on a model being built and trained using Keras (Tensorflow backend) and FPGA code generation and synthesis done by HLS4ML. 

## Instructions

Clone the repo and run the following commands within the directory:

`conda env create -f environment.yml`

`conda activate mnist_fpga`

`jupyter notebook`

At this point open the MNIST_FPGA_Implementation notebook. Note you have to adjust your PATH to point to your location of your installation of Vivado_HLS.

You can now run through the notebook.

## TODO:

* Compile model to FPGA (currently resource usage is larger than any FPGA I own)
* Write up a testbench for Vivado simulation
