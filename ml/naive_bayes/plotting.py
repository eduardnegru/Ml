import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import randint

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

acc = [
  57.6061,
  66.8024,
  90.9237
]

f1 = [

44.5702,
48.2361,
56.0749,

]

prec = [
  
66.8110,
62.9645,
55.6717,

]

recall = [
 
53.7607,
53.1965,
56.5871,

]

plt.xlabel("Tip set de date")
plt.ylabel("Procent")
ax.plot(["Supraeșantionat","Subeșantionat","Inițial"], acc, marker='o', label='Acuratețe')
ax.plot(["Supraeșantionat","Subeșantionat","Inițial"], f1, marker='o', label='F1')
ax.plot(["Supraeșantionat","Subeșantionat","Inițial"], prec, marker='o', label='Precizie')
ax.plot(["Supraeșantionat","Subeșantionat","Inițial"], recall, marker='o', label='Recall')
ax.legend()
fig.savefig('graph.png')
