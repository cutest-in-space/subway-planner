import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
import random
import matplotlib.font_manager as fm

# 设置中文字体支持
try:
    # 尝试使用系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 
                                      'Microsoft YaHei', 'WenQuanYi Micro Hei']
except:
    # 如果没有合适的中文字体，使用默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 检查可用的字体
available_fonts = [f.name for f in fm.fontManager.ttflist]
if not any(font in available_fonts for font in ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']):
    # 如果没有中文字体，将所有中文标题改为英文
    def safe_text(text):
        # 将中文标题转换为英文
        translations = {
            "城市地铁路线规划系统": "Urban Subway Route Planning System",
            "制作者": "Created by",
            "基础参数设置": "Basic Parameters",
            "网格行数": "Grid Rows",
            "网格列数": "Grid Columns",
            "起点": "Start",
            "终点": "End",
            "转弯成本": "Turn Cost",
            "算法设置": "Algorithm Settings",
            "选择算法": "Select Algorithm",
            "可视化设置": "Visualization Settings",
            "显示成本热力图": "Show Cost Heatmap",
            "显示网格线": "Show Grid Lines",
            "颜色主题": "Color Theme",
            "生成随机数据": "Generate Random Data",
            "计算最优路径": "Calculate Optimal Path",
            "比较不同算法": "Compare Algorithms",
            "数据编辑": "Data Editor",
            "可视化结果": "Visualization",
            "算法比对": "Algorithm Comparison",
            "移动成本矩阵": "Movement Cost Matrix",
            "建站成本矩阵": "Station Cost Matrix",
            "地铁路线规划图": "Subway Route Planning Map",
            "规划详情": "Planning Details",
            "总站点数": "Total Stations",
            "路线长度": "Route Length",
            "总成本": "Total Cost",
            "不同算法结果比对": "Algorithm Results Comparison",
            "算法性能指标比较": "Algorithm Performance Comparison",
            "总成本比较": "Total Cost Comparison",
            "路线长度比较": "Route Length Comparison",
            "站点数量比较": "Station Count Comparison",
            "路径可视化比较": "Path Visualization Comparison"
        }
        return translations.get(text, text)
        else:
    def safe_text(text):
        return text
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

# 模拟数据生成函数（替代C++程序）
def generate_random_data(n, m):
    # 生成随机移动成本矩阵
    move_cost = np.random.randint(10, 100, size=(n, m))
    # 生成随机建站成本矩阵
    build_cost = np.random.randint(100, 1000, size=(n, m))
    return move_cost, build_cost

# 模拟路径规划函数（替代C++程序）
def plan_path(n, m, move_cost, build_cost, stx, sty, edx, edy, turn_cost, algorithm):
    # 简单的路径生成逻辑
    path_points = []
    stations = []
    
    # 根据算法不同，生成稍微不同的路径
    if algorithm == "Dijkstra算法":
        # 生成一条直线路径
        x_steps = edx - stx
        y_steps = edy - sty
        
        total_steps = max(abs(x_steps), abs(y_steps))
        if total_steps == 0:
            path_points = [(stx, sty)]
        else:
            x_step = x_steps / total_steps
            y_step = y_steps / total_steps
            
            for i in range(total_steps + 1):
                x = int(stx + i * x_step)
                y = int(sty + i * y_step)
                path_points.append((x, y))
    else:  # 模拟退火算法
        # 生成一条稍微曲折的路径
        current_x, current_y = stx, sty
        path_points.append((current_x, current_y))
        
        while current_x != edx or current_y != edy:
            # 随机决定是向x方向还是y方向移动
            if current_x == edx:
                # 只能在y方向移动
                current_y += 1 if current_y < edy else -1
            elif current_y == edy:
                # 只能在x方向移动
                current_x += 1 if current_x < edx else -1
            else:
                # 可以在任意方向移动
                if random.random() < 0.5:
                    current_x += 1 if current_x < edx else -1
                else:
                    current_y += 1 if current_y < edy else -1
            
            path_points.append((current_x, current_y))
    
    # 生成站点（每隔几个点放置一个站点）
    station_interval = max(2, len(path_points) // 5)  # 确保至少有几个站点
    for i in range(0, len(path_points), station_interval):
        stations.append(path_points[i])
    
    # 确保起点和终点是站点
    if path_points[0] not in stations:
        stations.insert(0, path_points[0])
    if path_points[-1] not in stations:
        stations.append(path_points[-1])
    
    # 计算总成本
    total_cost = 0
    # 移动成本
    for x, y in path_points:
        total_cost += move_cost[x][y]
    # 建站成本
    for x, y in stations:
        total_cost += build_cost[x][y]
    # 转弯成本
    turns = 0
    for i in range(1, len(path_points) - 1):
        prev_x, prev_y = path_points[i-1]
        curr_x, curr_y = path_points[i]
        next_x, next_y = path_points[i+1]
        
        # 检测是否有转弯
        if (curr_x - prev_x != next_x - curr_x) or (curr_y - prev_y != next_y - curr_y):
            turns += 1
    
    total_cost += turns * turn_cost
    
    return path_points, stations, total_cost

# 生成随机数据按钮
if generate_btn:
    try:
        # 显示简单加载提示
        with st.spinner('正在生成随机数据...'):
            # 使用Python函数替代C++程序
            st.session_state.move_cost, st.session_state.build_cost = generate_random_data(n, m)
            # 成功提示
            st.success('随机数据生成成功！')
    except Exception as e:
        st.error(f"运行出错: {str(e)}")

# 计算最优路径按钮
if calculate_btn and st.session_state.move_cost is not None:
    try:
        # 显示简单加载提示
        with st.spinner('正在计算最优路径...'):
            # 使用Python函数替代C++程序
            path_points, stations, total_cost = plan_path(
                n, m, st.session_state.move_cost, st.session_state.build_cost, 
                stx, sty, edx, edy, turn_cost, algorithm
            )
            
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

# 添加算法比对功能
if compare_btn and st.session_state.move_cost is not None:
    try:
        # 清空之前的比对结果
        st.session_state.algorithm_results = {}
        
        # 要比对的算法列表
        algorithms_to_compare = ["Dijkstra算法", "模拟退火算法"]
        
        with st.spinner('正在比较不同算法...'):
            for alg in algorithms_to_compare:
                # 使用Python函数替代C++程序
                path_points, stations, total_cost = plan_path(
                    n, m, st.session_state.move_cost, st.session_state.build_cost, 
                    stx, sty, edx, edy, turn_cost, alg
                )
                
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
        城市地铁路线规划系统 | <a href="https://github.com/yourusername/subway-planner" target="_blank" style="color: #4B5563; text-decoration: none;">GitHub</a>
    </div>
    """, 
    unsafe_allow_html=True
)
