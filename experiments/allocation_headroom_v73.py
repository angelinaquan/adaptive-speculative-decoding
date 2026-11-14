"""Exact budgeted hindsight allocation. Acceptance reward only; not speed."""
def allocate(rewards,budget):
 # rewards[i][d-1] is the observed accepted-prefix length plus one.
 states={0:(0,[])}
 for row in rewards:
  next_states={}
  for used,(reward,widths) in states.items():
   for width,value in enumerate(row,1):
    cost=used+width
    if cost<=budget and (cost not in next_states or reward+value>next_states[cost][0]):
     next_states[cost]=(reward+value,widths+[width])
  states=next_states
 cost=max(states,key=lambda c:(states[c][0],-c));reward,widths=states[cost]
 return {'reward':reward,'draft_tokens':cost,'widths':widths}

if __name__=='__main__':
 import random,itertools
 rng=random.Random(18657)
 for n in [1,2,3]:
  for _ in range(40):
   rows=[[rng.randrange(d+1)+1 for d in range(1,9)] for i in range(n)]
   budget=3*n;got=allocate(rows,budget)
   possible=[(sum(rows[i][w-1] for i,w in enumerate(ws)),-sum(ws)) for ws in itertools.product(range(1,9),repeat=n) if sum(ws)<=budget]
   assert (got['reward'],-got['draft_tokens'])==max(possible)
 print('PASS:120 finite budget-allocation cases versus exhaustive enumeration')
