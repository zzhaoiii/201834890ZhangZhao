# 201834890ZhangZhao
Homework of Data Mining

2018-10-10
update 1：
1. vsm.py文件产生 01 型verctor space（运行贼慢！），并提供knn.py类库支持
2. knn.py现只能产生tf-idf权重的verctor space，现未实现分类，且运行太慢！
3. data/out/vsm-0.csv文件是vsm的0 1型vector space，但运行太慢，现只有第一个doc产生的demo

2018-10-11
update 2：
1. 更改词典生成规则，过滤doc中词频4以下token
2. 跑出01型vector space，但太大（500M），无法上传git上

2018-10-13
update 3：
1. 使用类库计算tf-idf，加快计算速度
2. 跑出tf-idf的vector space（1.4G）
3. 规范vsm.py中的代码

2018=10-18
update 4：
1. 规范化代码
2. 把预处理、生成字典等的过程产生的文件进行保存，方便后续调试，只需要读取csv文件即可
