---
title: Conda常用命令
tags:
  - 命令查询
abbrlink: 42466
date: 2023-08-17 12:26:17
---

## 1 查看版本
```shell
conda --version
```

## 2 虚拟环境
```shell
conda info -e # 显示所有虚拟环境
conda create -n your_env_name python=x.x  # 创建虚拟环境
```

## 3 使用代理
```shell
conda config --set proxy_servers.http http://127.0.0.1:7890
conda config --set proxy_servers.https https://127.0.0.1:7890
```