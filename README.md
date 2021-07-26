# ELM-super-fast
Python implementation of ELM - with optimized speed on MKL-based platforms; Described in conference paper: Radu Dogaru, Ioana Dogaru, "Optimization of extreme learning machines for big data applications using Python", COMM-2018
https://ieeexplore.ieee.org/document/8484265

ADVANTAGES: 
- allows quantization of input layer weights (in many cases 2 bits are enough) 
- allows quantization of output layer weights (in many cases 8 bits are enough)
- gives very good accuracies with tip=3 (absolute value) hidden layer 
Well suited for HW and other rezource-constrained machine learning implementations 

NOTE:  A much faster version with GPU support via CUPY library is available here: 
https://github.com/radu-dogaru/LB-CNN-compact-and-fast-binary-including-very-fast-ELM

Run directly in Google Colab: 
<a href="https://colab.research.google.com/github/radu-dogaru/LB-CNN-compact-and-fast-binary-including-very-fast-ELM/blob/main/LB_CNN_optimizer_trainer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
  
