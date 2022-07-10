rel_path=$(dirname "$0")                # relative
abs_path=$(cd "$rel_path" && pwd)       # absolutized and normalized

cd "$rel_path/.." && \
source polus_scripts/credentials.sh && \
SSHPASS=$SSHPASS sshpass -e sftp -oBatchMode=no -b - 23m_vel@calypso.gml-team.ru << !
   cd work/22k_koz/recognition-aware/datasets
   put $1
   bye
!
