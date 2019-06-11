# PCVch09-

python计算机视觉第九章实验-图像分割

## 图像分割

所谓图像分割指的是根据灰度、颜色、纹理和形状等特征把图像划分成若干互不交迭的区域，并使这些特征在同一区域内呈现出相似性，而在不同区域间呈现出明显的差异性。

**基于图割的分割方式**

此类方法把图像分割问题与图的最小割（min cut）问题相关联。首先将图像映射为带权无向图G=<V，E>，图中每个节点N∈V对应于图像中的每个像素，每条边∈E连接着一对相邻的像素，边的权值表示了相邻像素之间在灰度、颜色或纹理方面的非负相似度。而对图像的一个分割s就是对图的一个剪切，被分割的每个区域C∈S对应着图中的一个子图。而分割的最优原则就是使划分后的子图在内部保持相似度最大，而子图之间的相似度保持最小。基于图论的分割方法的本质就是移除特定的边，将图划分为若干子图从而实现分割。目前所了解到的基于图论的方法有**GraphCut**，GrabCut和Random Walk等。

今天将利用GraphCut方法来做两个简单的小例子

## 实例

我先将实现的代码附上来

```python
from pygraph.classes.digraph import digraph
from pygraph.algorithms.minmax import maximum_flow

gr = digraph()
gr.add_nodes([0,1,2,3])
gr.add_edge((0,1), wt=4)
gr.add_edge((1,2), wt=3)
gr.add_edge((2,3), wt=5)
gr.add_edge((0,2), wt=3)
gr.add_edge((1,3), wt=4)
flows,cuts = maximum_flow(gr, 0, 3)
print ('flow is:' , flows)
print ('cut is:' , cuts)
```

这里先是给出一个用python-graph工具包计算一幅较小的图的最大流/最小割的简单例子

在这段代码中，我们先创建了4个节点的有向图，4个节点的索引分别是0，1，2，3，然后用add_edge()增加每个边的权重。权重就是用来衡量边的最大流容量。以节点0为源点，3为汇点，计算最大流。

![](https://github.com/zengqq1997/PCVch09-/blob/master/fenge3.jpg)

上面图像展示的两行数字包含了流穿过每条边和每个节点的标记；0是包含图源点的部分，1是与汇点相连的节点

```python
# -*- coding: utf-8 -*-

from scipy.misc import imresize
from PCV.tools import graphcut
from PIL import Image
from numpy import *
from pylab import *

im = array(Image.open("empire.jpg"))
im = imresize(im, 0.07)
size = im.shape[:2]
print ("OK!!")

# add two rectangular training regions
labels = zeros(size)
labels[3:18, 3:18] = -1
labels[-18:-3, -18:-3] = 1
print ("OK!!")


# create graph
g = graphcut.build_bayes_graph(im, labels, kappa=1)

# cut the graph
res = graphcut.cut_graph(g, size)
print ("OK!!")


figure()
graphcut.show_labeling(im, labels)

figure()
imshow(res)
gray()
axis('off')

show()
```

我们利用imresize()函数使得图像小到我们适合我们的python—graph库，在这段代码中我们将图像统一缩小到原来图像尺寸的7%。图像分割后将结果和训练区域一起画出来。

![](https://github.com/zengqq1997/PCVch09-/blob/master/fenge1.jpg)

图上朴素贝叶斯分类器的模型

在这段代码中用来分割函数是graphcut.build_bayes_graph(im, labels, kappa=1)，其中从一个图像中的像素四邻域建立一个图，前景和背景，这边的前景和背景由我们的labels决定，并用朴素贝叶斯分类器建模。还有在这个函数中有个参数kappa决定了近邻像素间的相对权重。改变k值分割效果，随着k值的增大，分割界限将变得平滑，并且使得部分细节逐步丢失。

这里我将kappa设置为1和1.5得到如下图的结果。图一是kappa为1，图二是kappa为1.5

![](https://github.com/zengqq1997/PCVch09-/blob/master/fenge2.jpg)



![]()

