@echo off
echo 初始化Git仓库并推送代码到GitHub

REM 初始化Git仓库
git init

REM 添加所有文件
git add .

REM 创建初始提交
git commit -m "初始提交：城市地铁路线规划系统"

REM 添加远程仓库
set /p repo_url=请输入您的GitHub仓库URL（例如：https://github.com/yourusername/subway-planner.git）：
git remote add origin %repo_url%

REM 推送到GitHub
git push -u origin master

echo 代码已成功推送到GitHub！
echo 现在您可以在Streamlit Cloud上部署您的应用了。
pause 