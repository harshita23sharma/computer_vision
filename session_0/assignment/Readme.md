
# Assignment:


1. What are Channels and Kernels (according to EVA)?
2. Why should we (nearly) always use 3x3 kernels?
3. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
4. How are kernels initialized? 
5. What happens during the training of a DNN?

## Soultions

1. What are Channels and Kernels (according to EVA)?
    - One channel is like a container, containing similar kind of information. It can contain a fixed number of low level features which are combined in that particular layer as a single feature. At first level, one channel contains a single feature. Through subsequent layer, one channel can contain a combined low level information or features extracted from the previous layers as well. So after a few layers of convolution network, if we combine features from previous layers and see them as a single feature, that can be further contained with one channel to learn that particular feature.
    
    - Kernel on the other hand works to give/output that particular channel. It is a tool or matrix ( generally 3*3) that convolves over an image(matrix) or feature map to extract one feature. Results of the convolution with kernel results in channel containing feature. Kernel is also termed as feature extractor or filter. One kernel can only outputs one feature at a time and that feature can also be a combination of low level features from previous layers, now combined considered as a single feature.
    - e.g : Vertical kernel: 
    [
        [0,1,0],
        [0,1,0],
        [0,1,0]
    ]
    
    - e.g : Horizontal Kernel: 
    [
        [0,0,0],
        [1,1,1],
        [0,0,0]
    ]

    - For eg: Horizontal filter will filter out rest and give only horizontal lines from an image/channel.
    A vertical filter filters out rest and give only vertical lines from an image/channel.
    Now, in the next layer, we can use a kernel to extract L shape by convolving a kernel upon the channel/s of previous layer. We can also extract T shape by convolving a different kernel.

2. Why should we (nearly) always use 3x3 kernels?
    - 3x3 is smallest feasible kernel (which is odd) and infact it is better than other options we got like 5x5 or 7x7 or 9x9 etc. The reason is because of the receptive field. 
    So 3x3 kernel operates more locally to extract the feature. It is good because most of the features in the image are ususally local. Most of the features may be found in more than one place in an image. So using a single kernel of 3x3, we can extract different features in nearby parts of the image.

    - There are other advantages as well of using for say 2 3x3 kernels instead of 1 5x5, or 4 3x3 filters instead of 1 9x9 filter. The advantage of using 3x3 kernel is that it results in less number of parameters as compared to a bigger kernel size. Kernel values are learnt during backpropagation. so for eg: 9x9 filter will have 81 parameters to be learnt while training, whereas 4 3x3 kernel will have 4*9= 36 parameters only. So, it reduces the training computation as well.

4. How are kernels initialized? 
    - kernel is a matrix containing some values. For eg: 3x3 will have 9 numbers in a matrix form, which futher gets multiplied with other matrix in process of convolution. Initially the kernel values are initialized randomly, later on the values are learnt from backpropagation while training the convolution network end to end.
    Also, we don't want our model to be biased towards any particular feature while manually initializing the values.

5. What happens during the training of a DNN?
    - Deep Neural network comprises of one input layer, one output layer and multiple hidden layers. Each layer have got multiple neurons which stores some values or numbers or activations (between 0 and 1). We can also have some activation function assigned to some or all neurons in layers. The activation function modifies the value of neurons while training. Neurons in one layer are influenced by the neurons of the previous layers or layers on the top. There is weighted sum taken from the neurons in the previous layer and activation function is applied to keep the value of weighted sum in the acceptable range. So, during the forward pass in the network, activation function keeps doing the transformation on input received from the previous layers.

    - The way learning happens in the DNN is through backward pass. When we provide an input to the network, it makes the prediction in the forward pass. The prediction might be wrong or correct. The network is provided with this information feedback through backpropagation. In the back pass, the network is provided with the feedback to change it according to the actual ground truth in this supervised learning process. So the network learns by looking at the actual output and updates the weight inside the network according to an algorithm called back propagation. This method basically calculates the error function and takes gradient of that (slope) with respect to the weights and updates its weight according to that. The process of providing the input and updating the weights in this way is repeated several times with more and more data. This help the network update the weights such that when predicting on an unseen data, the weights learnt or the trained model gives the correct predictions. 

3. How many times do we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
    - It takes 99 convolution operation to reach from 199x199 to 1x1
        ```
        199 x 199  | 3x3 >  197 x 197
        197 x 197  | 3x3 >  195 x 195
        195 x 195  | 3x3 >  193 x 193
        193 x 193  | 3x3 >  191 x 191
        191 x 191  | 3x3 >  189 x 189
        189 x 189  | 3x3 >  187 x 187
        187 x 187  | 3x3 >  185 x 185
        185 x 185  | 3x3 >  183 x 183
        183 x 183  | 3x3 >  181 x 181
        181 x 181  | 3x3 >  179 x 179
        179 x 179  | 3x3 >  177 x 177
        177 x 177  | 3x3 >  175 x 175
        175 x 175  | 3x3 >  173 x 173
        173 x 173  | 3x3 >  171 x 171
        171 x 171  | 3x3 >  169 x 169
        169 x 169  | 3x3 >  167 x 167
        167 x 167  | 3x3 >  165 x 165
        165 x 165  | 3x3 >  163 x 163
        163 x 163  | 3x3 >  161 x 161
        161 x 161  | 3x3 >  159 x 159
        159 x 159  | 3x3 >  157 x 157
        157 x 157  | 3x3 >  155 x 155
        155 x 155  | 3x3 >  153 x 153
        153 x 153  | 3x3 >  151 x 151
        151 x 151  | 3x3 >  149 x 149
        149 x 149  | 3x3 >  147 x 147
        147 x 147  | 3x3 >  145 x 145
        145 x 145  | 3x3 >  143 x 143
        143 x 143  | 3x3 >  141 x 141
        141 x 141  | 3x3 >  139 x 139
        139 x 139  | 3x3 >  137 x 137
        137 x 137  | 3x3 >  135 x 135
        135 x 135  | 3x3 >  133 x 133
        133 x 133  | 3x3 >  131 x 131
        131 x 131  | 3x3 >  129 x 129
        129 x 129  | 3x3 >  127 x 127
        127 x 127  | 3x3 >  125 x 125
        125 x 125  | 3x3 >  123 x 123
        123 x 123  | 3x3 >  121 x 121
        121 x 121  | 3x3 >  119 x 119
        119 x 119  | 3x3 >  117 x 117
        117 x 117  | 3x3 >  115 x 115
        115 x 115  | 3x3 >  113 x 113
        113 x 113  | 3x3 >  111 x 111
        111 x 111  | 3x3 >  109 x 109
        109 x 109  | 3x3 >  107 x 107
        107 x 107  | 3x3 >  105 x 105
        105 x 105  | 3x3 >  103 x 103
        103 x 103  | 3x3 >  101 x 101
        101 x 101  | 3x3 >  99 x 99
        99 x 99  | 3x3 >  97 x 97
        97 x 97  | 3x3 >  95 x 95
        95 x 95  | 3x3 >  93 x 93
        93 x 93  | 3x3 >  91 x 91
        91 x 91  | 3x3 >  89 x 89
        89 x 89  | 3x3 >  87 x 87
        87 x 87  | 3x3 >  85 x 85
        85 x 85  | 3x3 >  83 x 83
        83 x 83  | 3x3 >  81 x 81
        81 x 81  | 3x3 >  79 x 79
        79 x 79  | 3x3 >  77 x 77
        77 x 77  | 3x3 >  75 x 75
        75 x 75  | 3x3 >  73 x 73
        73 x 73  | 3x3 >  71 x 71
        71 x 71  | 3x3 >  69 x 69
        69 x 69  | 3x3 >  67 x 67
        67 x 67  | 3x3 >  65 x 65
        65 x 65  | 3x3 >  63 x 63
        63 x 63  | 3x3 >  61 x 61
        61 x 61  | 3x3 >  59 x 59
        59 x 59  | 3x3 >  57 x 57
        57 x 57  | 3x3 >  55 x 55
        55 x 55  | 3x3 >  53 x 53
        53 x 53  | 3x3 >  51 x 51
        51 x 51  | 3x3 >  49 x 49
        49 x 49  | 3x3 >  47 x 47
        47 x 47  | 3x3 >  45 x 45
        45 x 45  | 3x3 >  43 x 43
        43 x 43  | 3x3 >  41 x 41
        41 x 41  | 3x3 >  39 x 39
        39 x 39  | 3x3 >  37 x 37
        37 x 37  | 3x3 >  35 x 35
        35 x 35  | 3x3 >  33 x 33
        33 x 33  | 3x3 >  31 x 31
        31 x 31  | 3x3 >  29 x 29
        29 x 29  | 3x3 >  27 x 27
        27 x 27  | 3x3 >  25 x 25
        25 x 25  | 3x3 >  23 x 23
        23 x 23  | 3x3 >  21 x 21
        21 x 21  | 3x3 >  19 x 19
        19 x 19  | 3x3 >  17 x 17
        17 x 17  | 3x3 >  15 x 15
        15 x 15  | 3x3 >  13 x 13
        13 x 13  | 3x3 >  11 x 11
        11 x 11  | 3x3 >  9 x 9
        9 x 9  | 3x3 >  7 x 7
        7 x 7  | 3x3 >  5 x 5
        5 x 5  | 3x3 >  3 x 3
        3 x 3  | 3x3 >  1 x 1
    ```

