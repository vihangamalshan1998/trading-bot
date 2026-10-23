@echo off
REM One-off: refits the brain's input scaling (switches the late inputs + replayed history on once 3,000
REM complete live records exist), saves it, then keeps training like MASTER_TRAINER. Close MASTER_TRAINER first.
cd /d "%~dp0"
call MASTER_TRAINER.bat --refit-feature-stats
