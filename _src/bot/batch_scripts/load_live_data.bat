@echo off
REM will start live loading function for pgsql 
cd C:\gh\trading-bot\_src\bot\
C:\gh\venv38\Scripts\python workflow_scripts.py wf__live_pg_write
pause