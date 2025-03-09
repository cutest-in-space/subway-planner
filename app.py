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

# Solution representation: path (continuous coordinates) and station build flags
class Solution:
    def __init__(self, path, built):
        self.path = path[:]    # List of (x, y) coordinates
        self.built = built[:]  # Corresponds to path, True means build a station at this node

def calc_cost(sol):
    """Cost function: Calculate the total cost of a solution"""
    total_cost = 0
    
    # Movement cost: accumulate the movement cost of each node on the path
    for x, y in sol.path:
        total_cost += sa_move_cost[x][y]
    
    # Station building cost
    for i, (x, y) in enumerate(sol.path):
        if sol.built[i]:
            total_cost += sa_build_cost[x][y]
    
    # Turn cost: for direction changes between consecutive steps (no turn cost for the first step)
    for i in range(2, len(sol.path)):
        prev_x, prev_y = sol.path[i-2]
        curr_x, curr_y = sol.path[i-1]
        next_x, next_y = sol.path[i]
        
        # Check if there's a direction change
        prev_dir = (curr_x - prev_x, curr_y - prev_y)
        next_dir = (next_x - curr_x, next_y - curr_y)
        if prev_dir != next_dir:
            total_cost += sa_turn_cost
    
    return total_cost

def is_simple(path):
    """Check if a path is simple (no self-intersections)"""
    visited = set()
    for p in path:
        if tuple(p) in visited:
            return False
        visited.add(tuple(p))
    return True

def generate_manhattan_path(s, t):
    """Generate a Manhattan path from s to t"""
    path = []
    x, y = s
    while x != t[0] or y != t[1]:
        path.append((x, y))
        if x < t[0]:
            x += 1
        elif x > t[0]:
            x -= 1
        elif y < t[1]:
            y += 1
        else:
            y -= 1
    path.append(t)
    return path

def neighbor(sol):
    """Generate a neighboring solution by making a small change"""
    # Deep copy the solution
    new_path = sol.path[:]
    new_built = sol.built[:]
    
    # Randomly choose an operation
    op = random.randint(0, 2)
    
    if op == 0:  # Modify station building
        if len(new_path) > 0:
            idx = random.randint(0, len(new_path) - 1)
            new_built[idx] = not new_built[idx]
    
    elif op == 1:  # Modify path - add a detour
        if len(new_path) > 1:
            idx = random.randint(0, len(new_path) - 2)
            x, y = new_path[idx]
            nx, ny = new_path[idx + 1]
            
            # Only consider direct neighbors
            if abs(nx - x) + abs(ny - y) == 1:
                # Try to add a detour through a neighboring point
                dx, dy = nx - x, ny - y
                if dx == 0:  # Moving vertically
                    if 0 <= x + 1 < sa_rows:
                        new_path.insert(idx + 1, (x + 1, y))
                        new_path.insert(idx + 2, (x + 1, ny))
                        new_built.insert(idx + 1, False)
                        new_built.insert(idx + 2, False)
                    elif 0 <= x - 1 < sa_rows:
                        new_path.insert(idx + 1, (x - 1, y))
                        new_path.insert(idx + 2, (x - 1, ny))
                        new_built.insert(idx + 1, False)
                        new_built.insert(idx + 2, False)
                else:  # Moving horizontally
                    if 0 <= y + 1 < sa_cols:
                        new_path.insert(idx + 1, (x, y + 1))
                        new_path.insert(idx + 2, (nx, y + 1))
                        new_built.insert(idx + 1, False)
                        new_built.insert(idx + 2, False)
                    elif 0 <= y - 1 < sa_cols:
                        new_path.insert(idx + 1, (x, y - 1))
                        new_path.insert(idx + 2, (nx, y - 1))
                        new_built.insert(idx + 1, False)
                        new_built.insert(idx + 2, False)
    
    elif op == 2:  # Remove a detour if possible
        if len(new_path) > 3:
            idx = random.randint(0, len(new_path) - 3)
            x1, y1 = new_path[idx]
            x2, y2 = new_path[idx + 1]
            x3, y3 = new_path[idx + 2]
            
            # Check if we can remove the middle point
            if abs(x3 - x1) + abs(y3 - y1) == 1:
                new_path.pop(idx + 1)
                new_built.pop(idx + 1)
    
    # Ensure the path still connects start and end
    if new_path[0] != (sa_start_x, sa_start_y) or new_path[-1] != (sa_end_x, sa_end_y):
        return Solution(sol.path, sol.built)
    
    return Solution(new_path, new_built)

def simulated_annealing_path_planning(move_cost_matrix, build_cost_matrix, stx, sty, edx, edy, turn_cost_value, 
                                     initial_temp=1000.0, cooling_rate=0.995, max_iterations=100000):
    """
    Simulated annealing algorithm for subway path planning
    
    Parameters:
    - move_cost_matrix: Cost matrix for moving through each cell
    - build_cost_matrix: Cost matrix for building a station at each cell
    - stx, sty: Start point coordinates
    - edx, edy: End point coordinates
    - turn_cost_value: Cost penalty for making a turn
    - initial_temp: Initial temperature for simulated annealing
    - cooling_rate: Rate at which temperature decreases
    - max_iterations: Maximum number of iterations
    
    Returns:
    - path_points: List of coordinates in the optimal path
    - stations: List of coordinates where stations are built
    - total_cost: Total cost of the solution
    """
    global sa_rows, sa_cols, sa_move_cost, sa_build_cost, sa_start_x, sa_start_y, sa_end_x, sa_end_y, sa_turn_cost
    
    # Set global variables for the algorithm
    sa_rows = len(move_cost_matrix)
    sa_cols = len(move_cost_matrix[0])
    sa_move_cost = move_cost_matrix
    sa_build_cost = build_cost_matrix
    sa_start_x, sa_start_y = stx, sty
    sa_end_x, sa_end_y = edx, edy
    sa_turn_cost = turn_cost_value
    
    # Initialize with a Manhattan path
    init_path = generate_manhattan_path((stx, sty), (edx, edy))
    
    # Initialize station building decisions (build at start and end, random for others)
    init_built = [False] * len(init_path)
    init_built[0] = True  # Always build at start
    init_built[-1] = True  # Always build at end
    
    # Randomly decide whether to build stations at intermediate points
    for i in range(1, len(init_path) - 1):
        init_built[i] = random.random() < 0.3  # 30% chance to build a station
    
    current_sol = Solution(init_path, init_built)
    best_sol = Solution(init_path, init_built)
    current_cost = calc_cost(current_sol)
    best_cost = current_cost
    
    # Simulated annealing parameters
    temp = initial_temp
    
    # Main simulated annealing loop
    for iteration in range(max_iterations):
        # Generate a neighboring solution
        new_sol = neighbor(current_sol)
        new_cost = calc_cost(new_sol)
        
        # Decide whether to accept the new solution
        cost_diff = new_cost - current_cost
        if cost_diff < 0 or random.random() < math.exp(-cost_diff / temp):
            current_sol = new_sol
            current_cost = new_cost
            
            # Update best solution if needed
            if current_cost < best_cost:
                best_sol = Solution(current_sol.path, current_sol.built)
                best_cost = current_cost
        
        # Cool down
        temp *= cooling_rate
        
        # Stop if temperature is too low
        if temp < 0.1:
            break
    
    # Extract the path points and stations from the best solution
    path_points = best_sol.path
    stations = [path_points[i] for i in range(len(path_points)) if best_sol.built[i]]
    
    return path_points, stations, best_cost

# 添加数据生成函数，替代datagen.py
def generate_data(n, m, stx, sty, edx, edy, turn_cost):
    """
    Generate random data for the subway planning problem
    
    Parameters:
    - n, m: Grid dimensions
    - stx, sty: Start point coordinates
    - edx, edy: End point coordinates
    - turn_cost: Cost penalty for making a turn
    
    Returns:
    - move_cost: Movement cost matrix
    - build_cost: Station building cost matrix
    """
    # Generate random movement costs (200-1000)
    move_cost = [[random.randint(200, 1000) for _ in range(m)] for _ in range(n)]
    
    # Generate random building costs (500-2000)
    build_cost = [[random.randint(500, 2000) for _ in range(m)] for _ in range(n)]
    
    # Make start and end points have lower costs to encourage building stations there
    move_cost[stx][sty] = max(50, move_cost[stx][sty] // 2)
    move_cost[edx][edy] = max(50, move_cost[edx][edy] // 2)
    build_cost[stx][sty] = max(100, build_cost[stx][sty] // 3)
    build_cost[edx][edy] = max(100, build_cost[edx][edy] // 3)
    
    return move_cost, build_cost

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
        self.direction = direction  # 0: none, 1: up, 2: right, 3: down, 4: left
        self.cost = cost
        self.path = path
        self.built_stations = built_stations
    
    def __lt__(self, other):
        return self.cost < other.cost

def planner_main():
    """Main function for the Dijkstra algorithm implementation"""
    # Read grid dimensions
    line = input().strip().split()
    n, m = int(line[0]), int(line[1])
    
    # Read start and end coordinates
    line = input().strip().split()
    stx, sty = int(line[0]), int(line[1])
    line = input().strip().split()
    edx, edy = int(line[0]), int(line[1])
    
    # Read turn cost
    turn_cost = int(input().strip())
    
    # Read movement cost matrix
    move_cost = []
    for i in range(n):
        line = input().strip().split()
        move_cost.append([int(x) for x in line])
    
    # Read station building cost matrix
    build_cost = []
    for i in range(n):
        line = input().strip().split()
        build_cost.append([int(x) for x in line])
    
    # Direction vectors: none, up, right, down, left
    dx = [0, -1, 0, 1, 0]
    dy = [0, 0, 1, 0, -1]
    
    # Initialize visited array and priority queue
    visited = [[False for _ in range(m)] for _ in range(n)]
    pq = []
    
    # Start node: always build a station at the start
    start_node = Node(stx, sty, 0, build_cost[stx][sty], [(stx, sty)], [(stx, sty)])
    heapq.heappush(pq, start_node)
    
    # Dijkstra's algorithm
    while pq:
        node = heapq.heappop(pq)
        x, y = node.x, node.y
        
        # If we've reached the end point, build a station and return the result
        if x == edx and y == edy:
            # Always build a station at the end
            if (edx, edy) not in node.built_stations:
                node.built_stations.append((edx, edy))
                node.cost += build_cost[edx][edy]
            
            # Output the result
            print(len(node.path))
            for px, py in node.path:
                print(f"{px} {py}")
            
            print(len(node.built_stations))
            for sx, sy in node.built_stations:
                print(f"{sx} {sy}")
            
            print(node.cost)
            return
        
        # Skip if already visited
        if visited[x][y]:
            continue
        
        visited[x][y] = True
        
        # Try all four directions
        for dir in range(1, 5):
            nx, ny = x + dx[dir], y + dy[dir]
            
            # Check if the new position is valid
            if 0 <= nx < n and 0 <= ny < m and not visited[nx][ny]:
                # Calculate the new cost
                new_cost = node.cost + move_cost[nx][ny]
                
                # Add turn cost if direction changes
                if node.direction != 0 and node.direction != dir:
                    new_cost += turn_cost
                
                # Create a new path and copy the built stations
                new_path = node.path.copy()
                new_path.append((nx, ny))
                new_built = node.built_stations.copy()
                
                # Create a new node
                new_node = Node(nx, ny, dir, new_cost, new_path, new_built)
                heapq.heappush(pq, new_node)
    
    # If we can't reach the end point
    print("No valid path found")

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
st.set_page_config(layout="wide", page_title="城市地铁路线规划系统", page_icon="🚇")

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
    st.title("🚇 城市地铁路线规划系统")
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
    with st.expander("🌐 基础参数设置", expanded=True):
        n = st.number_input("网格行数 (n)", min_value=2, value=10)
        m = st.number_input("网格列数 (m)", min_value=2, value=10)
        stx = st.number_input("起点X", min_value=0, max_value=n-1, value=0)
        sty = st.number_input("起点Y", min_value=0, max_value=m-1, value=0)
        edx = st.number_input("终点X", min_value=0, max_value=n-1, value=n-1)
        edy = st.number_input("终点Y", min_value=0, max_value=m-1, value=m-1)
        turn_cost = st.number_input("转弯成本", min_value=0, value=100)
    
    # 算法选择
    with st.expander("🧮 算法设置", expanded=True):
        algorithm = st.selectbox(
            "选择算法",
            ["Dijkstra算法", "模拟退火算法"]
        )
        
        # 算法参数设置
        if algorithm == "模拟退火算法":
            st.session_state.sa_initial_temp = st.slider("初始温度", 100.0, 2000.0, 1000.0, 100.0)
            st.session_state.sa_cooling_rate = st.slider("冷却率", 0.8, 0.999, 0.99, 0.001)
            st.session_state.sa_iterations = st.slider("最大迭代次数", 1000, 20000, 10000, 1000)
    
    # 可视化设置
    with st.expander("🎨 可视化设置", expanded=True):
        grid_size = st.slider("网格大小", 5, 20, 10)
        show_costs = st.checkbox("显示成本热力图", value=True)
        show_grid = st.checkbox("显示网格线", value=True)
        
        color_theme = st.selectbox(
            "颜色主题",
            ["蓝色主题", "绿色主题", "红色主题", "紫色主题"]
        )
        
        # 根据选择设置颜色
        if color_theme == "蓝色主题":
            line_color = "#1E40AF"
            station_color = "white"
            station_edge = "#1E3A8A"
            start_color = "green"
            end_color = "red"
        elif color_theme == "绿色主题":
            line_color = "#047857"
            station_color = "white"
            station_edge = "#065F46"
            start_color = "blue"
            end_color = "red"
        elif color_theme == "红色主题":
            line_color = "#B91C1C"
            station_color = "white"
            station_edge = "#991B1B"
            start_color = "green"
            end_color = "blue"
        else:  # 紫色主题
            line_color = "#7E22CE"
            station_color = "white"
            station_edge = "#6B21A8"
            start_color = "green"
            end_color = "red"
    
    # 数据生成和计算按钮
    st.markdown("### 🚀 操作")
    col1, col2 = st.columns(2)
    with col1:
        generate_btn = st.button("生成随机数据", use_container_width=True)
    with col2:
        calculate_btn = st.button("计算最优路径", use_container_width=True)
    
    # 添加算法比对按钮
    compare_btn = st.button("比较不同算法", use_container_width=True)

with right_col:
    # 显示和编辑数据
    if st.session_state.move_cost is not None:
        tabs = st.tabs(["📊 数据编辑", "🗺️ 可视化结果", "📈 算法比对"])
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("移动成本矩阵")
                move_df = pd.DataFrame(st.session_state.move_cost)
                edited_move_cost = st.data_editor(move_df, use_container_width=True)
                st.session_state.move_cost = edited_move_cost.values
            
            with col2:
                st.subheader("建站成本矩阵")
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
                    plt.colorbar(im, ax=ax, label='移动成本')
                
                # 绘制网格
                if show_grid:
                    ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
                
                # 设置网格大小
                ax.set_xticks(np.arange(-0.5, m, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
                
                # 绘制路径和站点
                path_x = [p[1] for p in st.session_state.path_points]
                path_y = [n-1-p[0] for p in st.session_state.path_points]
                ax.plot(path_x, path_y, '-', color=line_color, linewidth=3, zorder=2, label='地铁线路')
                
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
                                          label='地铁站点')
                
                # 起终点标记
                ax.scatter(sty, n-1-stx, color=start_color, s=250, marker='*', 
                          label='起点', zorder=4)
                ax.scatter(edy, n-1-edx, color=end_color, s=250, marker='*', 
                          label='终点', zorder=4)
                
                # 添加图例和标题
                handles, labels = ax.get_legend_handles_labels()
                handles.append(station_marker)
                labels.append('地铁站点')
                ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 1))
                
                ax.set_title("地铁路线规划图", pad=20, fontsize=16)
                
                # 显示图形
                st.pyplot(fig)
                
                # 显示详细信息
                st.markdown("### 📋 规划详情")
                
                # 使用自定义CSS样式的指标卡
                metric_html = f"""
                <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.stations)}</div>
                        <div class="metric-label">总站点数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{len(st.session_state.path_points)}</div>
                        <div class="metric-label">路线长度</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{st.session_state.total_cost:,}</div>
                        <div class="metric-label">总成本</div>
                    </div>
                </div>
                """
                st.markdown(metric_html, unsafe_allow_html=True)

        # 添加算法比对选项卡
        with tabs[2]:
            if len(st.session_state.algorithm_results) > 0:
                st.subheader("不同算法结果比对")
                
                # 创建比对表格
                compare_data = []
                for alg_name, result in st.session_state.algorithm_results.items():
                    compare_data.append({
                        "算法": alg_name,
                        "总成本": result["total_cost"],
                        "路线长度": len(result["path_points"]),
                        "站点数量": len(result["stations"])
                    })
                
                compare_df = pd.DataFrame(compare_data)
                st.dataframe(compare_df, use_container_width=True)
                
                # 创建三个横向排列的图表来比较不同指标
                st.subheader("算法性能指标比较")
                
                # 创建三列布局
                col1, col2, col3 = st.columns(3)
                
                algorithms = [data["算法"] for data in compare_data]
                
                # 1. 总成本比较图
                with col1:
                    fig1, ax1 = plt.subplots(figsize=(4, 3))
                    costs = [data["总成本"] for data in compare_data]
                    
                    bars1 = ax1.bar(algorithms, costs, color='#ff9999')
                    ax1.set_ylabel('总成本')
                    ax1.set_title('总成本比较', fontsize=10)
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
                    path_lengths = [data["路线长度"] for data in compare_data]
                    
                    bars2 = ax2.bar(algorithms, path_lengths, color='#66b3ff')
                    ax2.set_ylabel('路线长度')
                    ax2.set_title('路线长度比较', fontsize=10)
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
                    station_counts = [data["站点数量"] for data in compare_data]
                    
                    bars3 = ax3.bar(algorithms, station_counts, color='#99ff99')
                    ax3.set_ylabel('站点数量')
                    ax3.set_title('站点数量比较', fontsize=10)
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
                st.subheader("路径可视化比较")
                
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
                           linewidth=3, zorder=2, label=f'{alg_name}路线')
                    
                    # 绘制站点
                    station_x = [s[1] for s in result["stations"]]
                    station_y = [n-1-s[0] for s in result["stations"]]
                    
                    for x, y in zip(station_x, station_y):
                        circle = plt.Circle((x, y), 0.2, color=colors[i % len(colors)], 
                                          ec='white', lw=1, zorder=3, alpha=0.7)
                        ax.add_patch(circle)
                
                # 起终点标记
                ax.scatter(sty, n-1-stx, color='gold', s=250, marker='*', 
                          label='起点', zorder=4)
                ax.scatter(edy, n-1-edx, color='purple', s=250, marker='*', 
                          label='终点', zorder=4)
                
                ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                ax.set_title("不同算法路线比较", pad=20, fontsize=16)
                
                st.pyplot(fig)

# 修改生成随机数据按钮
if generate_btn:
    try:
        # 显示简单加载提示
        with st.spinner('正在生成随机数据...'):
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
                st.success('随机数据生成成功！')
            else:
                st.error("生成的数据格式不正确")
    except Exception as e:
        st.error(f"运行出错: {str(e)}")

# 修改计算最优路径按钮
if calculate_btn and st.session_state.move_cost is not None:
    try:
        # 显示简单加载提示
        with st.spinner('正在计算最优路径...'):
            # 构建输入数据
            move_cost_matrix = np.array(st.session_state.move_cost)
            build_cost_matrix = np.array(st.session_state.build_cost)
            
            # 根据选择的算法调用相应的Python模块
            if algorithm == "Dijkstra算法":
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
                st.success('最优路径计算成功！')
    except Exception as e:
        st.error(f"运行出错: {str(e)}")

# 修改算法比对功能
if compare_btn and st.session_state.move_cost is not None:
    try:
        # 清空之前的比对结果
        st.session_state.algorithm_results = {}
        
        # 要比对的算法列表
        algorithms_to_compare = ["Dijkstra算法", "模拟退火算法"]
        
        with st.spinner('正在比较不同算法...'):
            # 准备公共输入数据
            move_cost_matrix = np.array(st.session_state.move_cost)
            build_cost_matrix = np.array(st.session_state.build_cost)
            input_data = f"{n} {m}\n"
            input_data += "\n".join(" ".join(map(str, row)) for row in st.session_state.move_cost) + "\n"
            input_data += "\n".join(" ".join(map(str, row)) for row in st.session_state.build_cost) + "\n"
            input_data += f"{stx} {sty} {edx} {edy}\n{turn_cost}"
            
            for alg in algorithms_to_compare:
                if alg == "Dijkstra算法":
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
                st.success(f'成功比较了 {len(st.session_state.algorithm_results)} 种算法！请查看"算法比对"选项卡。')
            else:
                st.error("算法比对失败，未能获取有效结果。")
    except Exception as e:
        st.error(f"运行出错: {str(e)}")

# 添加页脚
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; padding: 10px 0;">
        城市地铁路线规划系统
    </div>
    """, 
    unsafe_allow_html=True
)
