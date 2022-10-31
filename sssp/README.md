# sssp

## 准备数据

### 数据格式

数据使用边表格式，每行的数据如下所示，文件后缀名为`.wel`
```
src dst weight
```
比如数据:
```
0 1 26
0 3 33
2 3 40
1 2 10
```

### 将数据转换成二进制格式

1. 编译`converter`工具：
```
cd tools && make
```
2. 转换数据格式：
```
./converter ../data/test.wel
```

## 运行示例

1. 编译sssp程序：

```
make
```

2. 运行示例：

```
./sssp 0 ./data/test.bwcsr
```
