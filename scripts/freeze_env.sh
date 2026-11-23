#!/usr/bin/env bash
# Run after a successful end-to-end run, then commit requirements.lock.txt
pip freeze | grep -v "^-e" > requirements.lock.txt
python -c "import sys,platform; print('python', sys.version.split()[0], platform.platform())" >> requirements.lock.txt.info
echo "wrote requirements.lock.txt"
