rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

cd "$rel_path"
while true; do
  if test -f "run.lock"; then           # file with job number exists
    num="$(cat run.lock)"
    jobs=$(bjobs)
    if [[ $jobs == *"$num"* ]]; then
      sleep 1                           # job is working
    else
      rm "run.lock"
    fi
  else                                  # file does not exist, launch training
    #echo -e "Job <123456> submitted to queue <normal>\nWaiting for job to finish" | sed -n 's/Job <\([0-9]*\)>.*/\1/p' > "run.lock"
    bash submit_net.sh | sed -n 's/Job <\([0-9]*\)>.*/\1/p' > "run.lock"
    num="$(cat run.lock)"
    echo "Submitted job <$num> at $(date)"
  fi
  sleep 10
done