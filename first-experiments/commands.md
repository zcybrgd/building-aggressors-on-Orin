La commande pour compiler le code CUDA est la suivante :

```bash 
nvcc -o main.exe main.cu
```

La commande de compilation pour conserver les fichiers intermédiaires comme le PTX est :

```bash
nvcc -lineinfo -keep -arch=sm_75 main.cu -o main.exe
```

Pour profiler l'exécution du programme CUDA, voir la concurrence sur le GPU à partir de Nsys et obtenir des statistiques, utilisez la commande suivante :

```bash
nsys profile --stats=true main.exe 
```

ça va génerer un fichier report.nsys-rep que vous pouvez ouvrir avec Nsight Systems pour analyser les performances et la concurrence des kernels CUDA.


ncu pour les métriques du cache L2 seulement pour seulement le --kernel-name victimKernel  : 
```bash
ncu --kernel-name victimKernel --metrics lts__t_sector_hit_rate,lts__t_sectors_lookup_miss,lts__t_requests_op_read_lookup_miss,dram__bytes_read,smsp__average_warp_latency_issue_stalled_long_scoreboard,sm_issue_active,smsp_inst_issued main.exe
```


.sum   = Total across all blocks/SMs for ONE kernel launch
.avg   = Average per unit (block, SM partition, or profiling pass)
.min/.max = Variation bounds (useful for detecting anomalies)
