# Source this file from the repository root before running any script.
export WORK="${WORK:-$(pwd)}"
export TMPDIR="$WORK/.tmp"
export HF_HOME="$WORK/.hf"
mkdir -p "$TMPDIR" "$WORK/logs" "$WORK/outputs" "$WORK/data"
