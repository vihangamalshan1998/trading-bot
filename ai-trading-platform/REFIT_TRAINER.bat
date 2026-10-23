@echo off
REM One-off refit: CONTINUES the live brain (models/production, not the latest work-in-progress candidate), refits its
REM input scaling with the late inputs + replayed history on (refuses to start below 3,000 graded complete records),
REM saves it, then keeps training like MASTER_TRAINER. For a brand-new brain instead, replace --from-champion with
REM --fresh-brain. Close MASTER_TRAINER first.
cd /d "%~dp0"
call MASTER_TRAINER.bat --refit-feature-stats --from-champion
