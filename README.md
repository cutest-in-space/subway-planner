# 城市地铁路线规划系统

这是一个基于Streamlit开发的城市地铁路线规划系统，支持使用Dijkstra算法和模拟退火算法进行路线优化。

## 功能特点

- 支持自定义网格大小和起终点位置
- 可以生成随机的移动成本和建站成本数据
- 提供两种算法选择：Dijkstra算法和模拟退火算法
- 可视化展示规划结果，包括路线图和站点分布
- 支持算法性能对比分析
- 提供多种可视化主题选择

## 在线访问

访问 [Streamlit Cloud](https://streamlit.io) 上的部署版本

## 本地运行

1. 克隆仓库：
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行应用：
```bash
streamlit run app.py
```

## 使用说明

1. 设置基础参数（网格大小、起终点位置等）
2. 生成随机数据或手动编辑数据
3. 选择算法并设置相关参数
4. 点击"计算最优路径"获取结果
5. 使用"比较不同算法"功能进行算法对比

## 技术栈

- Python 3.8+
- Streamlit
- NumPy
- Pandas
- Matplotlib

## 作者

陆冠宇小组 