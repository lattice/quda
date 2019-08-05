#pragma once


/**
 * sets the cpu affinity of the calling process to the affinity mask reported by nvidia-smi topo
 * Note that older driver versions might pin all mpi ranks to the same single conre instead of a range
 * @param  deviceid gpu to determine affinity for
 * @return          0 if numa affinity was set
 */
int setNumaAffinityNVML(int deviceid);
