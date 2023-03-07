upload:
	scp -r optimizer/ ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/
get_graph:
	scp ${USER}@chome.metz.supelec.fr:/usr/users/${GROUP}/${USER}/myGraph.png myGraph.png
