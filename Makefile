upload:
	scp -r optimizer/ ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/

LOGFILES = $(wildcard *.log)
get_logs:
	$(foreach logfile,$(LOGFILES),scp -r ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/$(logfile) $(logfile);)