import streamlit as st
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import io
from io import StringIO
import sys
import heapq
import random
import math

# Set up basic font and display settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# Add simulated annealing algorithm code
# Global variables: map size, movement cost, station building cost, start/end coordinates and turn cost
sa_rows = sa_cols = 0
sa_move_cost = []
sa_build_cost = []
sa_start_x = sa_start_y = sa_end_x = sa_end_y = 0
sa_turn_cost = 0

# 解的表示：路径（连续的坐标）和每个节点是否建站的标志
class Solution:
    def __init__(self, path, built):
        self.path = path[:]    # 列表，每个元素为 (x, y)
        self.built = built[:]  # 与 path 对应，True 表示该节点建站

def calc_cost(sol):
    """成本函数：计算一条解的总成本"""
    cost = 0
    # 移动成本：路径上每个节点的移动成本均要累加
    for (x, y) in sol.path:
        cost += sa_move_cost[x][y]
    # 建站成本
    for i in range(len(sol.path)):
        if sol.built[i]:
            cost += sa_build_cost[sol.path[i][0]][sol.path[i][1]]
    # 转向代价：对连续两步之间方向变化（第一步没有转向代价）
    if len(sol.path) >= 2:
        dx_prev = sol.path[1][0] - sol.path[0][0]
        dy_prev = sol.path[1][1] - sol.path[0][1]
        for i in range(2, len(sol.path)):
            dx_cur = sol.path[i][0] - sol.path[i - 1][0]
            dy_cur = sol.path[i][1] - sol.path[i - 1][1]
            if dx_cur != dx_prev or dy_cur != dy_prev:
                cost += sa_turn_cost
            dx_prev, dy_prev = dx_cur, dy_cur
    return cost

def is_simple(path):
    """辅助函数：判断路径是否简单（没有重复节点）"""
    seen = set()
    for p in path:
        if p in seen:
            return False
        seen.add(p)
    return True

def generate_manhattan_path(s, t):
    """
    生成两点之间的曼哈顿路径（随机打乱步序），保证从 s 到 t 的一个可行路径
    移动方向编码：0—上, 1—下, 2—左, 3—右
    """
    path = [s]
    dx = t[0] - s[0]
    dy = t[1] - s[1]
    moves = []
    if dx > 0:
        moves.extend([1] * dx)
    else:
        moves.extend([0] * (-dx))
    if dy > 0:
        moves.extend([3] * dy)
    else:
        moves.extend([2] * (-dy))
    random.shuffle(moves)
    cur_x, cur_y = s
    for move in moves:
        if move == 0:
            cur_x -= 1  # 上
        elif move == 1:
            cur_x += 1  # 下
        elif move == 2:
            cur_y -= 1  # 左
        elif move == 3:
            cur_y += 1  # 右
        path.append((cur_x, cur_y))
    return path

def neighbor(sol):
    """
    邻域操作：从当前解产生一个新解
    op=0：随机翻转一个非起点和终点的建站决策
    op=1：随机选择路径中的一段，重新生成这段子路径（保证路径简单）
    """
    newSol = Solution(sol.path, sol.built)
    op = random.randint(0, 1)
    if op == 0:
        # 随机选择一个非起点和终点节点翻转建站标记
        if len(newSol.path) > 2:
            idx = random.randint(1, len(newSol.path) - 2)
            newSol.built[idx] = not newSol.built[idx]
    else:
        # 修改路径结构：随机选择路径中的两个位置 i 和 j (1 <= i < j <= n-1)
        if len(newSol.path) > 3:
            i = random.randint(1, len(newSol.path) - 2)
            # j 在 [i+1, len(path)-1] 范围内
            if i < len(newSol.path) - 1:
                j = random.randint(i + 1, len(newSol.path) - 1)
                s_point = newSol.path[i]
                t_point = newSol.path[j]
                newSegment = generate_manhattan_path(s_point, t_point)
                # 构造候选路径：保留 [0, i] 段，接上新生成的子路径（去掉重复的起点和终点），再接上 [j, end] 段
                candidate = newSol.path[:i + 1]
                candidate_built = newSol.built[:i + 1]
                if len(newSegment) > 2:
                    for k in range(1, len(newSegment) - 1):
                        candidate.append(newSegment[k])
                        candidate_built.append(False)  # 新生成的点默认不建站
                candidate.extend(newSol.path[j:])
                candidate_built.extend(newSol.built[j:])
                # 若候选路径简单，则采用该修改；否则保持原解
                if is_simple(candidate):
                    newSol.path = candidate
                    newSol.built = candidate_built
    return newSol

def simulated_annealing_path_planning(move_cost_matrix, build_cost_matrix, stx, sty, edx, edy, turn_cost_value, 
                                     initial_temp=1000.0, cooling_rate=0.995, max_iterations=100000):
    """
    使用模拟退火算法进行路径规划
    
    参数:
    move_cost_matrix: 移动成本矩阵
    build_cost_matrix: 建站成本矩阵
    stx, sty: 起点坐标
    edx, edy: 终点坐标
    turn_cost_value: 转弯成本
    initial_temp: 初始温度
    cooling_rate: 冷却率
    max_iterations: 最大迭代次数
    
    返回:
    path_points: 路径点列表
    stations: 站点列表
    total_cost: 总成本
    """
    global sa_rows, sa_cols, sa_move_cost, sa_build_cost, sa_start_x, sa_start_y, sa_end_x, sa_end_y, sa_turn_cost
    
    # 设置全局变量
    sa_rows, sa_cols = move_cost_matrix.shape
    sa_move_cost = move_cost_matrix
    sa_build_cost = build_cost_matrix
    sa_start_x, sa_start_y = stx, sty
    sa_end_x, sa_end_y = edx, edy
    sa_turn_cost = turn_cost_value
    
    # 检查坐标有效性
    if not (0 <= sa_start_x < sa_rows and 0 <= sa_start_y < sa_cols and 0 <= sa_end_x < sa_rows and 0 <= sa_end_y < sa_cols):
        raise ValueError("Invalid coordinates.")
    
    random.seed()  # 使用当前时间作为随机种子
    
    # 初始解：采用曼哈顿路径（从起点到终点）
    initial_path = generate_manhattan_path((sa_start_x, sa_start_y), (sa_end_x, sa_end_y))
    # 优化初始解：起点和终点必须建站，其他点以 30% 概率建站
    initial_built = [False] * len(initial_path)
    initial_built[0] = True
    initial_built[-1] = True
    for i in range(1, len(initial_path) - 1):
        if random.random() < 0.3:
            initial_built[i] = True
    # 确保至少有4个站点（包括起点和终点）
    station_count = sum(1 for b in initial_built if b)
    if station_count < 4:
        non_station_indices = [i for i in range(1, len(initial_path) - 1) if not initial_built[i]]
        random.shuffle(non_station_indices)
        to_add = min(4 - station_count, len(non_station_indices))
        for i in range(to_add):
            initial_built[non_station_indices[i]] = True
    
    current = Solution(initial_path, initial_built)
    current_cost = calc_cost(current)
    best = Solution(current.path, current.built)
    best_cost = current_cost
    
    # 模拟退火参数设置
    T = initial_temp
    alpha = cooling_rate
    T_min = 1e-3
    iteration = 0
    
    # 模拟退火主循环
    while T > T_min and iteration < max_iterations:
        candidate = neighbor(current)
        # 确保起点和终点始终建站
        candidate.built[0] = True
        candidate.built[-1] = True
        
        candidate_cost = calc_cost(candidate)
        delta = candidate_cost - current_cost
        
        if delta < 0 or math.exp(-delta / T) > random.random():
            current = Solution(candidate.path, candidate.built)
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = Solution(candidate.path, candidate.built)
                best_cost = current_cost
        
        T *= alpha
        iteration += 1
        
        # 每1000次迭代检查一次站点数量，若站点太少则增加
        if iteration % 1000 == 0:
            station_count = sum(1 for b in current.built if b)
            if station_count < max(4, int(len(current.path) * 0.2)):
                non_station_indices = [i for i in range(1, len(current.path) - 1) if not current.built[i]]
                if non_station_indices:
                    random.shuffle(non_station_indices)
                    to_add = min(2, len(non_station_indices))  # 每次最多添加2个站点
                    for i in range(to_add):
                        current.built[non_station_indices[i]] = True
                    current_cost = calc_cost(current)
    
    # 提取结果
    path_points = best.path
    stations = [best.path[i] for i in range(len(best.path)) if best.built[i]]
    
    return path_points, stations, best_cost

# 添加数据生成函数，替代datagen.py
def generate_data(n, m, stx, sty, edx, edy, turn_cost):
    """
    生成随机数据，模拟原datagen.cpp的功能
    
    参数:
    n, m: 网格尺寸
    stx, sty: 起点坐标
    edx, edy: 终点坐标
    turn_cost: 转弯成本
    """
    random.seed()
    
    # 输出网格尺寸
    print(f"{n} {m}")
    
    # 生成移动成本矩阵
    for i in range(n):
        row = [random.randint(0, 1999) for _ in range(m)]
        print(" ".join(map(str, row)))
    
    # 生成建站成本矩阵
    for i in range(n):
        row = [random.randint(-400, 3599) for _ in range(m)]
        print(" ".join(map(str, row)))
    
    # 输出起点、终点和转弯成本
    print(f"{stx} {sty} {edx} {edy}")
    print(turn_cost)

# 添加planner.py的全部代码
# 定义方向：上、下、左、右
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]
dir_char = ['U', 'D', 'L', 'R']  # 用于输出（本代码中未直接使用）

# 用于输出标记，类似于 C++ 中的 ansf 数组（可选）
ansf = [[0] * 510 for _ in range(510)]

class Node:
    def __init__(self, x, y, direction, cost, path, built_stations):
        self.x = x
        self.y = y
        self.direction = direction  # 进入该节点的方向（0:上, 1:下, 2:左, 3:右）
        self.cost = cost            # 到达该节点的累计花费
        self.path = path[:]         # 路径记录，列表中存储 (x, y)
        self.built_stations = built_stations[:]  # 已建站点列表，列表中存储 (x, y)

    def __lt__(self, other):
        return self.cost < other.cost

def planner_main():  # 重命名为planner_main以保持与原来代码的一致性
    # 读入行数和列数
    rows, cols = map(int, sys.stdin.readline().split())

    # 读入移动花费和建站花费
    move_cost = [list(map(int, sys.stdin.readline().split())) for _ in range(rows)]
    build_cost = [list(map(int, sys.stdin.readline().split())) for _ in range(rows)]

    # 读入起点和终点坐标
    start_x, start_y, end_x, end_y = map(int, sys.stdin.readline().split())
    turn_cost = int(sys.stdin.readline())

    # 检查坐标合法性
    if not (0 <= start_x < rows and 0 <= start_y < cols and 0 <= end_x < rows and 0 <= end_y < cols):
        print("Invalid coordinates.")
        return

    # visited[x][y][direction][built] 表示在 (x,y) 以某个方向进入且是否建站的状态是否被访问过
    visited = [[[[False for _ in range(2)] for _ in range(4)] for _ in range(cols)] for _ in range(rows)]

    pq = []
    # 将四个初始方向的节点加入队列，并分别考虑不建站和建站两种情况
    for i in range(4):
        initial_path = [(start_x, start_y)]
        # 不建站
        heapq.heappush(pq, Node(start_x, start_y, i, move_cost[start_x][start_y], initial_path, []))
        # 建站
        heapq.heappush(pq, Node(start_x, start_y, i,
                                  move_cost[start_x][start_y] + build_cost[start_x][start_y],
                                  initial_path, [(start_x, start_y)]))

    best_node = Node(-1, -1, -1, float('inf'), [], [])

    # Dijkstra 算法主循环
    while pq:
        current = heapq.heappop(pq)
        x, y, direction, cost = current.x, current.y, current.direction, current.cost

        # 如果到达终点，则更新最优解，但不能立即退出，需搜索全局最优解
        if x == end_x and y == end_y:
            if cost < best_node.cost:
                best_node = current
            continue

        # 检查当前状态是否已访问：判断当前节点是否已经建站
        has_built = any(station[0] == x and station[1] == y for station in current.built_stations)
        if visited[x][y][direction][1 if has_built else 0]:
            continue
        visited[x][y][direction][1 if has_built else 0] = True

        # 尝试向四个方向移动
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]

            if 0 <= nx < rows and 0 <= ny < cols:
                # 检查下一个点是否已在当前路径中（防止环路）
                if (nx, ny) in current.path:
                    continue

                new_cost = cost + move_cost[nx][ny] + (0 if i == direction else turn_cost)
                next_path = current.path + [(nx, ny)]

                # 不建站，直接移动过去
                heapq.heappush(pq, Node(nx, ny, i, new_cost, next_path, current.built_stations))

                # 尝试在新位置建站，如果尚未在此建站
                if not any(station[0] == nx and station[1] == ny for station in current.built_stations):
                    next_built_stations = current.built_stations + [(nx, ny)]
                    new_cost_build = new_cost + build_cost[nx][ny]
                    heapq.heappush(pq, Node(nx, ny, i, new_cost_build, next_path, next_built_stations))

    # 输出结果
    if best_node.cost == float('inf'):
        print("No path found.")
    else:
        print("Minimum cost:", best_node.cost)
        print("Path:", " -> ".join(f"({x}, {y})" for x, y in best_node.path))
        # 同时更新 ansf 数组，1 表示路径，2 表示建站
        for x, y in best_node.path:
            ansf[x][y] = 1
        print("Build Station:")
        if not best_node.built_stations:
            print("No station built.")
        else:
            for x, y in best_node.built_stations:
                print(f"({x}, {y})")
                ansf[x][y] = 2

# Set Chinese font support - now using English instead
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化session_state变量
if 'move_cost' not in st.session_state:
    st.session_state.move_cost = None
if 'build_cost' not in st.session_state:
    st.session_state.build_cost = None
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = None
if 'path_points' not in st.session_state:
    st.session_state.path_points = None
if 'stations' not in st.session_state:
    st.session_state.stations = None
# 添加算法比对结果存储
if 'algorithm_results' not in st.session_state:
    st.session_state.algorithm_results = {}
# 初始化模拟退火参数
if 'sa_initial_temp' not in st.session_state:
    st.session_state.sa_initial_temp = 1000.0
if 'sa_cooling_rate' not in st.session_state:
    st.session_state.sa_cooling_rate = 0.99
if 'sa_iterations' not in st.session_state:
    st.session_state.sa_iterations = 10000

# 页面配置
st.set_page_config(layout="wide", page_title="Urban Subway Planning System", page_icon="🚇")

# 添加CSS样式
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2E4A9A;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #6B7280;
    }
</style>
""", unsafe_allow_html=True)

# 页面标题和制作者信息
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🚇 Urban Subway Planning System")
with col2:
    st.markdown(
        """
        <div style="text-align: right; padding-top: 20px;">
            <span style="color: #6B7280; font-size: 1rem;">
                制作者: <span style="color: #1E3A8A; font-weight: bold;">陆冠宇小组</span>
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )
st.markdown("---")

# 创建两列布局
left_col, right_col = st.columns([1, 2])

with left_col:
    # 创建参数输入区
    with st.expander("🌐 Basic Parameter Settings", expanded=True):
        n = st.number_input("Grid Rows (n)", min_value=2, value=10)
        m = st.number_input("Grid Columns (m)", min_value=2, value=10)
        stx = st.number_input("Start X", min_value=0, max_value=n-1, value=0)
        sty = st.number_input("Start Y", min_value=0, max_value=m-1, value=0)
        edx = st.number_input("End X", min_value=0, max_value=n-1, value=n-1)
        edy = st.number_input("End Y", min_value=0, max_value=m-1, value=m-1)
        turn_cost = st.number_input("Turn Cost", min_value=0, value=100)
    
    # 算法选择
    with st.expander("🧮 Algorithm Settings", expanded=True):
        algorithm = st.selectbox(
            "Select Algorithm",
            ["Dijkstra Algorithm", "Simulated Annealing Algorithm"]
        )
        
        # 算法参数设置
        if algorithm == "Simulated Annealing Algorithm":
            st.session_state.sa_initial_temp = st.slider("Initial Temperature", 100.0, 2000.0, 1000.0, 100.0)
            st.session_state.sa_cooling_rate = st.slider("Cooling Rate", 0.8, 0.999, 0.99, 0.001)
            st.session_state.sa_iterations = st.slider("Maximum Iterations", 1000, 20000, 10000, 1000)
    
    # 可视化设置
    with st.expander("🎨 Visualization Settings", expanded=True):
        grid_size = st.slider("Grid Size", 5, 20, 10)
        show_costs = st.checkbox("Show Cost Heat Map", value=True)
        show_grid = st.checkbox("Show Grid Lines", value=True)
        
        color_theme = st.selectbox(
            "Color Theme",
            ["Blue Theme", "Green Theme", "Red Theme", "Purple Theme"]
        )
        
        # 根据选择设置颜色
        if color_theme == "Blue Theme":
            line_color = "#1E40AF"
            station_color = "white"
            station_edge = "#1E3A8A"
            start_color = "green"
            end_color = "red"
        elif color_theme == "Green Theme":
            line_color = "#047857"
            station_color = "white"
            station_edge = "#065F46"
            start_color = "blue"
            end_color = "red"
        elif color_theme == "Red Theme":
            line_color = "#B91C1C"
            station_color = "white"
            station_edge = "#991B1B"
            start_color = "green"
            end_color = "blue"
        else:  # Purple Theme
            line_color = "#7E22CE"
            station_color = "white"
            station_edge = "#6B21A8"
            start_color = "green"
            end_color = "red"
    
    # 数据生成和计算按钮
    st.markdown("### 🚀 Operations")
    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button("Generate Random Data", use_container_width=True)
    with col2:
        calculate_btn = st.button("Calculate Optimal Path", use_container_width=True)
    
    # 添加算法比对按钮
    compare_btn = st.button("Compare Different Algorithms", use_container_width=True)

with right_col:
    # 显示和编辑数据
    if st.session_state.move_cost is not None:
        tabs = st.tabs(["📊 Data Editing", "🗺️ Visualization Results", "📈 Algorithm Comparison"])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Movement Cost Matrix")
                move_df = pd.DataFrame(st.session_state.move_cost)
                edited_move_cost = st.data_editor(move_df, use_container_width=True)
                st.session_state.move_cost = edited_move_cost.values
            
            with col2:
                st.subheader("Station Building Cost Matrix")
                build_df = pd.DataFrame(st.session_state.build_cost)
                edited_build_cost = st.data_editor(build_df, use_container_width=True)
                st.session_state.build_cost = edited_build_cost.values
        
        with tabs[1]:
            if st.session_state.path_points is not None:
                # 使用matplotlib绘制路线图
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 设置样式
                plt.style.use('ggplot')
                
                # 绘制热力图显示成本
                if show_costs:
                    im = ax.imshow(st.session_state.move_cost, cmap='YlOrRd', alpha=0.3)
                    plt.colorbar(im, ax=ax, label='Movement Cost')
                
                # 绘制网格
                if show_grid:
                    ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                
                # 设置网格大小
                ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                
                # 绘制路径和站点
                path_x = [p[1] for p in st.session_state.path_points]
                path_y = [n-1-p[0] for p in st.session_state.path_points]
                ax.plot(path_x, path_y, '-', color=line_color, linewidth=3, zorder=2, label='Subway Line')
                
                # 绘制站点
                station_x = [s[1] for s in st.session_state.stations]
                station_y = [n-1-s[0] for s in st.session_state.stations]
                
                # 使用更美观的站点标记
                for x, y in zip(station_x, station_y):
                    # 绘制站点底色
                    circle = Circle((x, y), 0.3, color=station_color, ec=station_edge, lw=2, zorder=3)
                    ax.add_patch(circle)
                
                # 添加站点图例
                station_marker = plt.Line2D([], [], marker='o', color=station_color, markerfacecolor=station_color,
                                          markeredgecolor=station_edge, markersize=15, linestyle='None',
                                          label='Subway Stations')
                
                # 起终点标记
                ax.scatter(sty, n-1-stx, color=start_color, s=250, marker='*', 
                          label='Start', zorder=4)
                ax.scatter(edy, n-1-edx, color=end_color, s=250, marker='*', 
                          label='End', zorder=4)
                
                # 添加图例和标题
                handles, labels = ax.get_legend_handles_labels()
                handles.append(station_marker)
                labels.append('Subway Stations')
                ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
                
                ax.set_title("Subway Line Planning Map", pad=20, fontsize=16)
                
                # 显示图形
                st.pyplot(fig)
                
                # 显示详细信息
                st.markdown("### 📋 Planning Details")
                
                # 使用自定义CSS样式的指标卡
                metric_html = f"""
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.stations)}</div>
                        <div class="metric-label">Total Station Count</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.path_points)}</div>
                        <div class="metric-label">Path Length</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{st.session_state.total_cost:,}</div>
                        <div class="metric-label">Total Cost</div>
                    </div>
                </div>
                """
                st.markdown(metric_html, unsafe_allow_html=True)

        # 添加算法比对选项卡
        with tabs[2]:
            if len(st.session_state.algorithm_results) > 0:
                st.subheader("Algorithm Comparison")
                
                # 创建比对表格
                compare_data = []
                for alg_name, result in st.session_state.algorithm_results.items():
                    compare_data.append({
                        "Algorithm": alg_name,
                        "Total Cost": result["total_cost"],
                        "Path Length": len(result["path_points"]),
                        "Station Count": len(result["stations"])
                    })
                
                compare_df = pd.DataFrame(compare_data)
                st.dataframe(compare_df, use_container_width=True)
                
                # 创建三个横向排列的图表来比较不同指标
                st.subheader("Algorithm Performance Metric Comparison")
                
                # 创建三列布局
                col1, col2, col3 = st.columns(3)
                
                algorithms = [data["Algorithm"] for data in compare_data]
                
                # 1. 总成本比较图
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    costs = [data["Total Cost"] for data in compare_data]
                    
                    bars1 = ax1.bar(algorithms, costs, color='#ff9999')
                    ax1.set_ylabel('Total Cost')
                    ax1.set_title('Total Cost Comparison', fontsize=10)
                    # 添加数据标签
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height):,}',
                                ha='center', va='bottom', rotation=0, fontsize=8)
                    
                    # 调整x轴标签大小
                    ax1.tick_params(axis='x', labelsize=8)
                    ax1.tick_params(axis='y', labelsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                # 2. 路线长度比较图
                with col2:
                    fig2, ax2 = plt.subplots(figsize=(4, 3))
                    path_lengths = [data["Path Length"] for data in compare_data]
                    
                    bars2 = ax2.bar(algorithms, path_lengths, color='#66b3ff')
                    ax2.set_ylabel('Path Length')
                    ax2.set_title('Path Length Comparison', fontsize=10)
                    # 添加数据标签
                    for bar in bars2:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}',
                                ha='center', va='bottom', rotation=0, fontsize=8)
                    
                    # 调整x轴标签大小
                    ax2.tick_params(axis='x', labelsize=8)
                    ax2.tick_params(axis='y', labelsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                # 3. 站点数量比较图
                with col3:
                    fig3, ax3 = plt.subplots(figsize=(4, 3))
                    station_counts = [data["Station Count"] for data in compare_data]
                    
                    bars3 = ax3.bar(algorithms, station_counts, color='#99ff99')
                    ax3.set_ylabel('Station Count')
                    ax3.set_title('Station Count Comparison', fontsize=10)
                    # 添加数据标签
                    for bar in bars3:
                        height = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}',
                                ha='center', va='bottom', rotation=0, fontsize=8)
                    
                    # 调整x轴标签大小
                    ax3.tick_params(axis='x', labelsize=8)
                    ax3.tick_params(axis='y', labelsize=8)
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                # 绘制不同算法的路径比较
                st.subheader("Path Visualization Comparison")
                
                fig, ax = plt.subplots(figsize=(12, 8))
                plt.style.use('ggplot')
                
                # 绘制网格
                ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                
                # 为每个算法选择不同的颜色
                colors = ['#ff6666', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0']
                
                # 绘制每个算法的路径
                for i, (alg_name, result) in enumerate(st.session_state.algorithm_results.items()):
                    path_x = [p[1] for p in result["path_points"]]
                    path_y = [n-1-p[0] for p in result["path_points"]]
                    ax.plot(path_x, path_y, '-', color=colors[i % len(colors)], 
                           linewidth=3, zorder=2, label=f'{alg_name} Path')
                    
                    # 绘制站点
                    station_x = [s[1] for s in result["stations"]]
                    station_y = [n-1-s[0] for s in result["stations"]]
                    
                    for x, y in zip(station_x, station_y):
                        circle = plt.Circle((x, y), 0.2, color=colors[i % len(colors)], 
                                          ec='white', lw=1, zorder=3, alpha=0.7)
                        ax.add_patch(circle)
                
                # 起终点标记
                ax.scatter(sty, n-1-stx, color='gold', s=250, marker='*', 
                          label='Start', zorder=4)
                ax.scatter(edy, n-1-edx, color='purple', s=250, marker='*', 
                          label='End', zorder=4)
                
                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                ax.set_title("Different Algorithm Path Comparison", pad=20, fontsize=16)
                
                st.pyplot(fig)

# 修改生成随机数据按钮
if generate_btn:
    try:
        # 显示简单加载提示
        with st.spinner('Generating Random Data...'):
            # 使用Python模块代替C++程序
            import sys
            from io import StringIO
            
            # 重定向标准输出以捕获输出
            old_stdout = sys.stdout
            redirected_output = StringIO()
            sys.stdout = redirected_output
            
            # 调用我们的内置函数，而不是datagen模块
            generate_data(n, m, stx, sty, edx, edy, turn_cost)
            
            # 恢复标准输出
            sys.stdout = old_stdout
            
            # 处理输出数据
            data = redirected_output.getvalue().strip().split("\n")
            
            if len(data) >= 2*n + 1:  # 验证输出数据完整性
                # 跳过第一行（n m）
                move_cost_data = data[1:n+1]  # 取n行移动成本数据
                build_cost_data = data[n+1:2*n+1]  # 取n行建站成本数据
                
                # 解析每行数据，并只取前m个数
                st.session_state.move_cost = np.array([
                    list(map(int, row.split()))[:m] for row in move_cost_data
                ])
                st.session_state.build_cost = np.array([
                    list(map(int, row.split()))[:m] for row in build_cost_data
                ])
                
                # 成功提示
                st.success('Random Data Generation Successful!')
            else:
                st.error("生成的数据格式不正确")
    except Exception as e:
        st.error(f"运行出错: {str(e)}")

# 修改计算最优路径按钮
if calculate_btn and st.session_state.move_cost is not None:
    try:
        # 显示简单加载提示
        with st.spinner('Calculating Optimal Path...'):
            # 构建输入数据
            move_cost_matrix = np.array(st.session_state.move_cost)
            build_cost_matrix = np.array(st.session_state.build_cost)
            
            # 根据选择的算法调用相应的Python模块
            if algorithm == "Dijkstra Algorithm":
                # 准备输入数据
                input_data = f"{n} {m}\n"
                input_data += "\n".join(" ".join(map(str, row)) for row in st.session_state.move_cost) + "\n"
                input_data += "\n".join(" ".join(map(str, row)) for row in st.session_state.build_cost) + "\n"
                input_data += f"{stx} {sty} {edx} {edy}\n{turn_cost}"
                
                # 重定向标准输入和输出以捕获结果
                old_stdin = sys.stdin  # 保存原始标准输入
                old_stdout = sys.stdout  # 保存原始标准输出
                sys.stdin = StringIO(input_data)
                redirected_output = StringIO()
                sys.stdout = redirected_output
                
                # 调用 planner 模块
                planner_main()
                
                # 恢复标准输入和输出
                sys.stdin = old_stdin
                sys.stdout = old_stdout
                
                # 处理输出结果
                output = redirected_output.getvalue()
            else:  # 模拟退火算法
                # 直接调用函数接口而不是通过stdin/stdout
                path_points, stations, total_cost = simulated_annealing_path_planning(
                    move_cost_matrix, 
                    build_cost_matrix, 
                    stx, sty, edx, edy, turn_cost,
                    st.session_state.sa_initial_temp, 
                    st.session_state.sa_cooling_rate, 
                    st.session_state.sa_iterations
                )
                
                # 构造输出结果字符串，格式与C++版本一致
                output = f"Minimum cost: {total_cost}\n"
                output += f"Path: {' -> '.join([f'({x}, {y})' for x, y in path_points])}\n"
                output += "Build Station:\n"
                for x, y in stations:
                    output += f"({x}, {y})\n"
            
            # 解析输出结果
            output_lines = output.strip().split('\n')
            path_points = []
            stations = []
            total_cost = None
            
            # 解析文本结果
            reading_stations = False
            for i, line in enumerate(output_lines):
                if line.startswith("Minimum cost:"):
                    total_cost = int(line.split(":")[1].strip())
                elif line.startswith("Path:"):
                    # 解析路径坐标
                    coords = line[6:].strip()
                    path_points = eval("[" + coords.replace("->", ",") + "]")
                elif line.startswith("Build Station:"):
                    reading_stations = True
                    continue
                elif reading_stations and line.startswith("("):
                    # 解析车站坐标
                    coords = line.strip()
                    x, y = map(int, coords.strip("()").split(","))
                    stations.append((x, y))
            
            if total_cost is None:
                st.error("无法获取总成本")
            else:
                # 保存计算结果到session_state
                st.session_state.path_points = path_points
                st.session_state.stations = stations
                st.session_state.total_cost = total_cost
                
                # 保存当前算法结果
                st.session_state.algorithm_results[algorithm] = {
                    "total_cost": total_cost,
                    "path_points": path_points,
                    "stations": stations
                }
                
                # 成功提示
                st.success('Optimal Path Calculation Successful!')
    except Exception as e:
        st.error(f"运行出错: {str(e)}")

# 修改算法比对功能
if compare_btn and st.session_state.move_cost is not None:
    try:
        # 清空之前的比对结果
        st.session_state.algorithm_results = {}
        
        # 要比对的算法列表
        algorithms_to_compare = ["Dijkstra Algorithm", "Simulated Annealing Algorithm"]
        
        with st.spinner('Comparing Different Algorithms...'):
            # 准备公共输入数据
            move_cost_matrix = np.array(st.session_state.move_cost)
            build_cost_matrix = np.array(st.session_state.build_cost)
            input_data = f"{n} {m}\n"
            input_data += "\n".join(" ".join(map(str, row)) for row in st.session_state.move_cost) + "\n"
            input_data += "\n".join(" ".join(map(str, row)) for row in st.session_state.build_cost) + "\n"
            input_data += f"{stx} {sty} {edx} {edy}\n{turn_cost}"
            
            for alg in algorithms_to_compare:
                if alg == "Dijkstra Algorithm":
                    # 重定向标准输入和输出以捕获结果
                    old_stdin = sys.stdin  # 保存原始标准输入
                    old_stdout = sys.stdout  # 保存原始标准输出
                    sys.stdin = StringIO(input_data)
                    redirected_output = StringIO()
                    sys.stdout = redirected_output
                    
                    # 调用 planner 模块
                    planner_main()
                    
                    # 恢复标准输入和输出
                    sys.stdin = old_stdin
                    sys.stdout = old_stdout
                    
                    # 处理输出结果
                    output = redirected_output.getvalue()
                else:  # 模拟退火算法
                    # 直接调用函数接口
                    path_points, stations, total_cost = simulated_annealing_path_planning(
                        move_cost_matrix, 
                        build_cost_matrix, 
                        stx, sty, edx, edy, turn_cost,
                        st.session_state.sa_initial_temp, 
                        st.session_state.sa_cooling_rate, 
                        st.session_state.sa_iterations
                    )
                    
                    # 构造输出结果字符串
                    output = f"Minimum cost: {total_cost}\n"
                    output += f"Path: {' -> '.join([f'({x}, {y})' for x, y in path_points])}\n"
                    output += "Build Station:\n"
                    for x, y in stations:
                        output += f"({x}, {y})\n"
                
                # 解析输出结果
                output_lines = output.strip().split('\n')
                path_points = []
                stations = []
                total_cost = None
                
                # 解析文本结果
                reading_stations = False
                for i, line in enumerate(output_lines):
                    if line.startswith("Minimum cost:"):
                        total_cost = int(line.split(":")[1].strip())
                    elif line.startswith("Path:"):
                        # 解析路径坐标
                        coords = line[6:].strip()
                        path_points = eval("[" + coords.replace("->", ",") + "]")
                    elif line.startswith("Build Station:"):
                        reading_stations = True
                        continue
                    elif reading_stations and line.startswith("("):
                        # 解析车站坐标
                        coords = line.strip()
                        x, y = map(int, coords.strip("()").split(","))
                        stations.append((x, y))
                
                if total_cost is not None:
                    # 保存算法结果
                    st.session_state.algorithm_results[alg] = {
                        "total_cost": total_cost,
                        "path_points": path_points,
                        "stations": stations
                    }
            
            if len(st.session_state.algorithm_results) > 0:
                st.success(f'Successfully compared {len(st.session_state.algorithm_results)} algorithms! Please check the "Algorithm Comparison" tab.')
            else:
                st.error("Algorithm comparison failed, no valid results obtained.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; padding: 10px 0;">
        Urban Subway Planning System
    </div>
    """, 
    unsafe_allow_html=True
)
