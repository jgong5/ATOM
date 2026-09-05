cd /workspace/ATOM
export HF_HUB_OFFLINE=1
python -m pytest tests/compass/ -q 2>&1 | tail -12
