import json,random
from pathlib import Path
from retrieval_v94 import lookup,accepted
assert lookup([1,2,3,4,1,2],2,2)==([3,4],2)
assert lookup([1,2,3],2,2)==(None,0)
assert accepted([1,2,3],[1,4,3])==2
rng=random.Random(94)
for _ in range(500):
 prefix=[rng.randrange(4) for _ in range(rng.randrange(5,60))];w=rng.randrange(1,9);draft,n=lookup(prefix,w,2)
 matches=[]
 for size in range(2,min(32,len(prefix)-w)+1):
  for end in range(size,len(prefix)-w+1):
   if prefix[end-size:end]==prefix[-size:]:matches.append((size,end,prefix[end:end+w]))
 expected=max(matches,key=lambda x:(x[0],x[1])) if matches else None
 assert (draft,n)==((expected[2],expected[0]) if expected else (None,0))
out={'passed':True,'brute_force_cases':500,'checks':['Longest suffix and newest occurrence match exhaustive enumeration','Every continuation is fully inside observed prefix','Known lookup and acceptance examples'],'scope':'CPU lookup correctness only; not serving or lossless-generation integration'}
Path(__file__).with_name('retrieval_v94_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
