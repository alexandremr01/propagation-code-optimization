upload:
	scp -r optimizer/ ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/
get_log:
	scp -r ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/myLog.log myLog.log
