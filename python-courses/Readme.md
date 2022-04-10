# Main Resources



[Practical Python Course](./dabeaz/)

[Udemy-bootcamp](./udemy-bootcamp)


Textbooks:

[Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)

[Python Crash Course, Second Edition](https://ehmatthes.github.io/pcc_2e/regular_index/)


# Misc. Notes
- In contrast to `Dict()`, `OrderedDict()` preserves the order in which the keys are inserted. (matters when iterating on it)

- Set priority of a process

```python
"""
Set the priority of the process to high in Windows:
"""
# Found at
# https://stackoverflow.com/questions/1023038/change-process-priority-in-python-cross-platform
import win32api, win32process, win32con

pid = win32api.GetCurrentProcessId()
handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)
```