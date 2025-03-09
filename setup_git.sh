#!/bin/bash

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 创建初始提交
git commit -m "初始提交：城市地铁路线规划系统"

# 添加远程仓库（请替换为您的GitHub仓库URL）
echo "请输入您的GitHub仓库URL（例如：https://github.com/yourusername/subway-planner.git）："
read repo_url
git remote add origin $repo_url

# 推送到GitHub
git push -u origin master

echo "代码已成功推送到GitHub！"
echo "现在您可以在Streamlit Cloud上部署您的应用了。" 