import subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
for role,rd in [('ar',-1),('fixed7_greedy',0)]:subprocess.run([sys.executable,str(r/'benchmark_prefix_v52.py'),'--role',role,'--round',str(rd)],check=True)
