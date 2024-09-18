---
title: Markdown语法
abbrlink: 22648
date: 2023-08-31 23:48:55
mathjax: true
tags:
  - 命令查询
---

## 数学公式

### 矩阵相关

1. 不带括号的普通矩阵

    ```latex
    \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
    \end{matrix}
    ```

    $$
    \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix}
    $$

2. 带中括号的矩阵
   
    ```latex
    \left[
     \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix} 
    \right]
    ```
    
    $$
    \left[
     \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix} 
    \right]
    $$
    
3. 带大括号的矩阵
   
    ```latex
    \left\{
     \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix} 
    \right\}
    ```
    
    
    $$
    \left\{
     \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix} 
    \right\}
    $$
    
4. 矩阵前加参数
   
    ```latex
    A=
    \left\{
     \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix} 
    \right\}
    ```
    
    
    $$
    A=
    \left\{
     \begin{matrix}
       a & b & c & d & e \\
       f & g & h & i & j \\
       k & l & m & n & o \\
       p & q & r & s & t
      \end{matrix} 
    \right\}
    $$
    
5. 矩阵中间有省略号
   
    ```latex
    A=
    \left\{
     \begin{matrix}
       a & b & \cdots & e \\
       f & g & \cdots & j \\
       \vdots & \vdots & \ddots & \vdots \\
       p & q & \cdots & t
      \end{matrix} 
    \right\}
    ```
    
    
    $$
    A=
    \left\{
     \begin{matrix}
       a & b & \cdots & e \\
       f & g & \cdots & j \\
       \vdots & \vdots & \ddots & \vdots \\
       p & q & \cdots & t
      \end{matrix} 
    \right\}
    $$
    
6. 矩阵中间加横线
   
    ```latex
    A=
    \left\{
     \begin{array}{cccc|c}
         a & b & c & d & e \\
         f & g & h & i & j \\
         k & l & m & n & o \\
         p & q & r & s & t
      \end{array} 
    \right\}
    ```
    
    
    $$
    A=
    \left\{
     \begin{array}{cccc|c}
         a & b & c & d & e \\
         f & g & h & i & j \\
         k & l & m & n & o \\
         p & q & r & s & t
      \end{array} 
    \right\}
    $$

## 字体

| 写法                                  | 展示                                    |
| ------------------------------------- | --------------------------------------- |
| ABCDEFGHIJKLMNOPQRSTUVWXYZ            | $ABCDEFGHIJKLMNOPQRSTUVWXYZ$            |
| \mathbb{ABCDEFGHIJKLMNOPQRSTUVWXYZ}   | $\mathbb{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$   |
| \mathbf{ABCDEFGHIJKLMNOPQRSTUVWXYZ}   | $\mathbf{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$   |
| \mathtt{ABCDEFGHIJKLMNOPQRSTUVWXYZ}   | $\mathtt{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$   |
| \mathrm{ABCDEFGHIJKLMNOPQRSTUVWXYZ}   | $\mathrm{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$   |
| \mathsf{ABCDEFGHIJKLMNOPQRSTUVWXYZ}   | $\mathsf{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$   |
| \mathcal{ABCDEFGHIJKLMNOPQRSTUVWXYZ}  | $\mathcal{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  |
| \mathscr{ABCDEFGHIJKLMNOPQRSTUVWXYZ}  | $\mathscr{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$  |
| \mathfrak{ABCDEFGHIJKLMNOPQRSTUVWXYZ} | $\mathfrak{ABCDEFGHIJKLMNOPQRSTUVWXYZ}$ |

## 括号

| 写法                                      |                    展示                    |
| :---------------------------------------- | :----------------------------------------: |
| \left( \frac{a}{b} \right)                |        $\left( \frac{a}{b} \right)$        |
| \left[ \frac{a}{b} \right]                |        $\left[ \frac{a}{b} \right]$        |
| \left \langle \frac{a}{b} \right \rlangle | $\left \langle \frac{a}{b} \right \rangle$ |
| \left\| \frac{a}{b} \right\|              |        $\left| \frac{a}{b} \right|$        |
| \left \| \frac{a}{b} \right \|            |      $\left \|\frac{a}{b} \right \|$       |
| \left \lfoor \frac{a}{b} \right \rfloor   | $\left \lfloor \frac{a}{b} \right \rfloor$ |
| \left \lceil \frac{a}{b} \right \rceil    |  $\left \lceil \frac{a}{b} \right \rceil$  |

