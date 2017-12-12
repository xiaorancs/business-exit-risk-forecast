# [企业经营退出风险预测](http://www.datafountain.cn/#/competitions/271/intro)

+ 题目思路  
    [!思路图片](./企业风险预测.png)
+ model:    Xgboost
+ 数据处理： pandas

+ 文件夹结构介绍

    data/alldata/： 存放所有的得到的数据文件     
    data/public/: 题目给定的原始数据     
    (https://pan.baidu.com/s/1nuJNz9B) 
    提取密码：t2ek   
    运行之前，请建立对应的文件夹，并导入数据。   
    model/：运行的model文件   
    feature/: 提取特征的py文件     
    saveModel/: 保存model，可以不使用       
    stack/: stacking特征的py文件     
    xresult/：存放输出结果的文件

+ 排名

    初赛： 43 / 1192 (571个队伍)     
    复赛： 25 / 1192 (569个队伍)

注意：
如果使用python2运行出现错误，请在Python文件头加上：
```
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
``
