upload:
	scp -r optimizer/ ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/

get_logs:
	scp '${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/*.log' ./