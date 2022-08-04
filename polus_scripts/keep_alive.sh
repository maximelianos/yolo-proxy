rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

cd "$rel_path"
while true; do
  if test -f "run.lock"; then         # file exists, wait
    num="$(cat run.lock)"
    echo $num
    echo $(bjobs)
    if [[ $(bjobs) == *"$num"* ]]; then
      echo "It's there!"
    fi
    sleep 1
  else                                # file does not exist, launch training
    echo "Submit job at $(date)"
    bash sub_test.sh | sed 's/Job <\([0-9]*\)>.*/\1/' > "run.lock"
  fi
done