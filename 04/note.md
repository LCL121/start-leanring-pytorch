1.3是0维，但是[1.3]是1维，长度为1的Tensor

## dimension
dimension = 0 ===> Loss

dimension = 1 ===> Bias(偏置)
                   Linear input
                   
dimension = 2 ===> Linear input batch

dimension = 3 ===> RNN input batch

dimension = 4 ===> CNN: [b, c, h, w] [张数, 颜色, 高度, 宽度]
     
                   
## 我理解的.shape/.size()/.dim()：

.shape/.size() ===> 表示的是数据的形状，如[2, 3]：2行3列

.dim() ===> 表示的是数据的维度

