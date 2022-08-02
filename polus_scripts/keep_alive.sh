rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

while true; do
    if test -f "run.lock"; then         # file exists, wait
        sleep 1
    else                                # file does not exist, launch training
        echo "Submit job at $(date)"
        bash submit_net.sh
        touch "run.lock"
    fi
done