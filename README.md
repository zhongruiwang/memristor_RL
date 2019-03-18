# memristor RL

The MATLAB code for the memristor Q-network based reinforcement learning, for
Wang, Z. et al. Reinforcement learning with analogue memristor arrays, Nature Electronics (https://dx.doi.org/10.1038/s41928-019-0221-6)

# Contributions

Authors(Alphabetic) : Daniel Belkin, Can Li, Wenhao Song, Zhongrui Wang

Advisors: Prof. J. Joshua Yang (Email: jjyang at umass dot edu, Website: http://www.ecs.umass.edu/ece/jjyang/); Prof. Qiangfei Xia (Email: qxia at umass dot edu, Website: http://nano.ecs.umass.edu); 


# SYSTEM REQUIREMENT

The codes have been tested on Mathworks Matlab R2017b.

# DEMO

Run the script "main_cartpole.m" for the cart-pole problem.
Run the script "main_mountaincar.m" for the mountain car problem.
The codes that interface the hardware is not provided. 

# INSTRUCTION

The real_array2 backend are dependent on the measurement system and hardware. 
The demo runs a multilayer perceptron Q-network. For construction of other advanced Q-network, please use the framework https://github.com/zhongruiwang/memristorCNN.
The experimental data will be provided upon reasonable request.

# LICENSE

Copyright (c) 2018, 
Department of Electrical and Computer Engineering, University of Massachusetts Amherst
All rights reserved.
                      
LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
