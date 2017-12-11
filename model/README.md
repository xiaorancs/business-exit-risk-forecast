# model
文件的介绍

1. [1_model_xgb.py](./1_model_xgb.py)

    + 模型的原型，使用单一的随机化种子seed，进行预测。

2. [3_model_prov.py](./3_model_prov.py)

    + 模型的原型，使用单一的随机化种子seed，进行预测。但是对不同的省份PROV进行分开训练和预测。

3. [model_origin.py](./model_origin.py)

    + 模型的原型，使用三个不同的随机化种子seed，进行预测，进行结果平均。

4. [model_avg_prov.py](./model_avg_prov.py)

    + 模型的原型，使用3个不同的随机化种子seed，进行预测。但是对不同的省份PROV进行分开训练和预测，对三个seed的结果进行平均。

5. [keras.py](./keras.py)
    
    +   使用keras搭建的神经网络进行预测，（结果没有xgb理想）。
6. [avg_result.py](./avg_result.py)

    + 平均结果文件。

7. [result_alpha.py](./result_alpha.py)

    + 加权平均结果文件。

8. [model_help.py](./model_help.py)

    + 模型测试代码。
