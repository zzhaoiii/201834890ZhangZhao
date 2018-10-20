# 201834890ZhangZhao
Homework of Data Mining

## Final product description 
> 1. 预处理过程：token->normalization（去特殊字符、小写、判断是否英语单词）->Stemming->Stopword
> 2. 构造词典：去除频率4以下token
> 3. 实现01型与tf-idf权重型space vector
> 4. 实现通过计数法、1/d\*d 权重法进行KNN分类
> 5. 准确率：   

 |   实现        |  准确率  |   K值  |
 |   :----:      | :----:  | :----: |
 |   01型+计数法    |  0.729  |   30   |
 | 01型+权重法    |  0.753  |   42   |
 | tf-idf+计数法  |  0.733  |   40   |
 | tf-idf+权重法  |  0.742  |   40   | 

</div>

----------

### 2018-10-20 update 5：
1. 实现KNN，对划分20%数据集做测试集，准确率0.753
2. 分别实现通过01型、tf-idf权重型vector space进行KNN分类
3. 分别实现通过计数法、1/d\*d 权重法进行分类。

----------

### 2018-10-18 update 4：
1. 规范化代码
2. 把预处理、生成字典等的过程产生的文件进行保存，方便后续调试，只需要读取csv文件即可

----------

### 2018-10-13 update 3：
1. 使用类库计算tf-idf，加快计算速度
2. 跑出tf-idf的vector space（1.4G）
3. 规范vsm.py中的代码

----------

### 2018-10-11 update 2：
1. 更改词典生成规则，过滤docs中词频4以下token
2. 跑出01型vector space，但太大（500M），无法上传git上

----------

### 2018-10-10 update 1：
1. vsm.py文件产生 01 型verctor space（运行贼慢！），并提供knn.py类库支持
2. knn.py现只能产生tf-idf权重的verctor space，现未实现分类，且运行太慢！
3. data/out/vsm-0.csv文件是vsm的0 1型vector space，但运行太慢，现只有第一个doc产生的demo
