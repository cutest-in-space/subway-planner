# 城市地铁路线规划系统 - Streamlit Cloud部署指南

本指南将帮助您将城市地铁路线规划系统部署到Streamlit Cloud，使其可以在互联网上访问。

## 部署步骤

### 1. 准备GitHub仓库

1. 登录您的GitHub账号（如果没有，请先在[GitHub](https://github.com)注册）
2. 创建一个新的仓库，例如命名为"subway-planner"
3. 将本地项目文件推送到GitHub仓库：
   ```bash
   git init
   git add .
   git commit -m "初始提交：城市地铁路线规划系统"
   git remote add origin https://github.com/cutest-in-space/subway-planner
   git push -u origin main
   ```

### 2. 注册Streamlit Cloud账号

1. 访问[Streamlit Cloud](https://share.streamlit.io/)
2. 点击"Continue to sign-in"
3. 选择"Continue with GitHub"（推荐）或使用其他方式注册
4. 按照提示完成账号创建和GitHub账号关联

### 3. 部署应用

1. 在Streamlit Cloud控制台中，点击右上角的"New app"按钮
2. 在"Repository"下拉菜单中选择您刚刚创建的仓库（subway-planner）
3. 在"Branch"字段中选择"main"（或您的默认分支）
4. 在"Main file path"字段中输入"app.py"（您的主应用文件）
5. 点击"Deploy"按钮开始部署

### 4. 查看部署进度

部署过程中，您可以点击右下角的"Manage app"查看部署日志和进度。部署完成后，您将获得一个可以访问的URL，例如：`https://yourusername-subway-planner-app.streamlit.app`

### 5. 自定义应用URL（可选）

1. 在应用部署完成后，点击"Settings"
2. 在"Rename"字段中输入您想要的应用名称
3. 点击"Save"保存更改

## 常见问题及解决方案

### 依赖包安装问题

确保您的项目根目录中包含以下文件：

1. **requirements.txt**：列出所有Python依赖包
   ```
   streamlit==1.32.0
   numpy==1.24.3
   pandas==2.0.3
   matplotlib==3.7.2
   ```

2. **packages.txt**（如果需要系统级依赖）：
   ```
   ffmpeg
   libsm6
   libxext6
   build-essential
   python3-dev
   ```

### 常见错误及解决方法

1. **ImportError: libGL.so.1: cannot open shared object file**
   - 解决方法：在packages.txt中添加`ffmpeg libsm6 libxext6`

2. **NameError: name 'LOADER_DIR' is not defined**
   - 解决方法：在packages.txt中添加`ffmpeg`

3. **应用部署成功但无法访问**
   - 检查应用日志，查看是否有运行时错误
   - 确认app.py中的代码能在本地正常运行

## 更新应用

当您对代码进行修改后，只需将更改推送到GitHub仓库，Streamlit Cloud将自动重新部署您的应用：

```bash
git add .
git commit -m "更新：描述您的更改"
git push
```

## 注意事项

1. Streamlit Cloud是免费服务，但有一定的资源限制
2. 应用在一段时间不活动后会进入休眠状态，再次访问时需要几秒钟启动时间
3. 确保您的代码不包含敏感信息或密钥，因为部署在Streamlit Cloud上的应用是公开可访问的

祝您部署顺利！如有任何问题，可以参考[Streamlit官方文档](https://docs.streamlit.io/deploy/streamlit-community-cloud)获取更多帮助。 