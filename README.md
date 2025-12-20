# oplib
## op list
- Gemm
- Reduce
- Multi-Head Attention
- Vector add
- TODO

## backend 
- [x] torch
- [x] cuda
- [x] triton
- [ ] cutile
- [ ] tilelang

# Quick start
install requirement
```python
# pip
pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# uv
uv pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

install ops
```python
python setup.py install
```
run ops
```python
python run.py
```
results:
![描述文字](./assert//result.png)