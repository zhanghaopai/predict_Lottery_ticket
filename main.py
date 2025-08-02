#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：predict_Lottery_ticket 
@File    ：main
@Author  ：zhanghaopai@outlook.com
@Date    ：2025/8/2 14:58 
'''
import get_data
import run_predict
import run_train_model

if __name__ == "__main__":
    name = "ssq"
    # 获取数据
    get_data.run(name=name)
    run_train_model.run(name=name, train_test_split=0.8)
    run_predict.run(name=name)
    exit(0)
