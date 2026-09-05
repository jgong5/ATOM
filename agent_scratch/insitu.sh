cd /workspace/ATOM
export HF_HUB_OFFLINE=1
rm -rf compass_ops/prof && mkdir -p compass_ops/prof
python agent_scratch/profile_step.py --model Qwen/Qwen3-0.6B \
  --torch-profiler-dir compass_ops/prof \
  --num-prompts 4 --max-tokens 8 --prompt-tokens 64
echo "### INSITU DONE rc=$?"
