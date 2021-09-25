See also `file:///E:/THESIS/Implements/anote_important_notes.htm`



## GPU memory Usage/information

`> nvidia-smi`

To refresh it every 1 second in windows, `> nvidia-smi.exe -l 1`

In Linux, `watch -n 1 nvidia-smi`

`nvidia-smi --format=csv --query-gpu=power.draw,utilization.gpu,fan.speed,temperature.gpu`

**Display available memory on Ubuntu Linux**

In Linux terminal, `free -m`

`grep MemTotal /proc/meminfo` 

`$ cat /proc/meminfo`