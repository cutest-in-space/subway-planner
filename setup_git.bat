@echo off
chcp 65001

REM 获取批处理文件所在目录
cd /d "%~dp0"

echo [初始化Git仓库并推送代码到GitHub]
echo [当前工作目录: %cd%]

REM 添加所有文件
git add .

REM 创建提交
git commit -m "Update numpy version for Python 3.12 compatibility"

REM 推送到GitHub
git push origin master

echo [代码已成功推送到GitHub！]
echo [现在您可以在Streamlit Cloud上部署您的应用了]
pause 