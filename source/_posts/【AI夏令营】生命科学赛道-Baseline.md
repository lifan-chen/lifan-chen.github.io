---
title: 【AI夏令营】生命科学赛道-Baseline
categories:
  - Note
abbrlink: 60176
date: 2023-08-17 23:40:12
tags:
---

## 1 配置阿里云DSW环境

[阿里云机器学习Pai-DSW服务器部署教程](https://qwosdw576oc.feishu.cn/docx/NajfdyJm3oripXxrPFFczjSon4z)

## 2 下载数据

* [安装 ossutil 工具](https://help.aliyun.com/zh/oss/developer-reference/install-ossutil?spm=a2c22.12281978.0.0.48a3558ctL6RrI)
ossutil 是 OSS 的命令行管理工具，支持 Windows、Linux、macOS、ARM 系统。可以通过 ossutil 提供的方便、简洁、丰富的 Bucket 和 Object 命令管理自己的 OSS。
```shell
sudo -v ; curl https://gosspublic.alicdn.com/ossutil/install.sh | sudo bash
```
* 分别复制训练集和测试集的 ossutil 内网命令至 DSW 的 Terminal，即可下载

## 3 解压数据
```shell
# 解压训练集的压缩包文件
!unzip ai4bio_trainset.zip

# 解压测试集的压缩包文件
!unzip ai4bio_testset_final.zip
```

## 4 运行 life-baseline.ipynb
* 将 baseline 代码上传至 DSW
* 运行代码

## 5 提交结果
* 下载 baseline 运行后生成的文件 submit_2.txt
* 在比赛界面提交结果文件
* 在 我的成绩 界面查看成绩
