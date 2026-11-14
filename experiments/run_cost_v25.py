import subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
# Fixed order selected before any v25 results. Balanced confirmation is separate.
for method in ['fixed4','ar','fixed1','measured','fixed6','fixed2','original','fixed7','fixed3','fixed5']:
 subprocess.run([sys.executable,str(root/'benchmark_cost_v25.py'),'--method',method],check=True)
print('ALL_V25_COMPLETE',flush=True)
