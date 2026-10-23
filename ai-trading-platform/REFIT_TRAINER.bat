@echo off
REM One-off refit: starts a FRESH brain, fits its input scaling with the late inputs + replayed history on
REM (refuses to start below 3,000 graded complete records), saves it, then keeps training like MASTER_TRAINER.
REM The live brain keeps trading until the new one wins a tournament. Close MASTER_TRAINER first.
cd /d "%~dp0"
call MASTER_TRAINER.bat --refit-feature-stats --fresh-brain
