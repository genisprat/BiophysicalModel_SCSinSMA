# BiophysicalModel_SCSinSMA
 Biophysical Model use in Prat-Ortega et al 2024


This software has been developped in python 3.9 and NEURON 8.1.0.
Instructions to installed NEURON can be found in their wbsite: https://www.neuron.yale.edu/neuron/.
Before using the software, modfiles need to be compiled as explained in https://www.neuron.yale.edu/neuron/faq/mac


The Demo file simulates a population of Supraspinal inputs, a population of sensory afferent recruited by SCS and a SMA-affected pool of motoneurons for 5 seconds.
The output are the raster plot for each population. Data is saved in a pickle. The expected run time it depends on the parameters. For the default parameters, it should be around 30 seconds in a normal desktop computer.


![output_Demo_Supraspinal](https://github.com/genisprat/BiophysicalModel_SCSinSMA/assets/22342465/ac11cee0-b63b-487c-8053-b12e81f9db5c)
![output_Demo_SCS](https://github.com/genisprat/BiophysicalModel_SCSinSMA/assets/22342465/e4b3b48d-6733-4029-8035-37cebfd4a96c)
![output_Demo_MN](https://github.com/genisprat/BiophysicalModel_SCSinSMA/assets/22342465/62cbb586-2802-4530-a9c8-fa14879e3814)
