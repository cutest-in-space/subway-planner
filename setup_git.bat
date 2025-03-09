@echo off
chcp 65001

REM 获取批处理文件所在目录
cd /d "%~dp0"

echo [初始化Git仓库并推送代码到GitHub]
echo [当前工作目录: %cd%]

REM 初始化Git仓库
git init

REM 添加所有文件
git add .

REM 创建初始提交
git commit -m "Initial commit: Metro System Planning"

REM 添加远程仓库
set /p repo_url=Please input your GitHub repository URL (e.g.: https://github.com/yourusername/subway-planner.git): 
git remote add origin %repo_url%

REM 推送到GitHub
git push -u origin master

echo [代码已成功推送到GitHub！]
echo [现在您可以在Streamlit Cloud上部署您的应用了]
pause 