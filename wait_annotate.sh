#!/bin/bash
cd ~/SWAY
while pgrep -f "main.py build-all" >/dev/null; do sleep 60; done
echo "=== build finished; annotating ==="
for c in b2 b4 b6; do
  last=$(ls -d results/build_artifacts/$c/iter_* 2>/dev/null | sort -V | tail -1)
  echo "########## $c ($last) ##########"
  python tools/annotate_transcripts.py --glob "$last/transcript_[0-2].json" --cell $c
done
echo ALL_DONE
