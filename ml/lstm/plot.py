import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import randint

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)


acc1 = [

79.9685,
83.0895,
85.7909,
88.3043,
90.4650,
92.1870
]

f11 = [
57.5241,
59.2236,
60.6304,
61.7605,
62.1543,
60.5849
]

acc3 = [
81.9473,
83.4051,
84.7503,
86.1182,
87.7029,
89.7157

]

f12 = [
58.2706,
59.0141,
59.6653,
60.3015,
60.9832,
61.4354
]

thresholds = [0.4, 0.5, 0.6, 0.7,0.8, 0.9]

plt.xlabel("Prag")
plt.ylabel("Procent")
ax.plot(thresholds, acc1, marker='o', label='Acuratețe LSTM 1')
ax.plot(thresholds, f12, marker='o', label='F1 LSTM 1')

ax.plot(thresholds, acc2, marker='o', label='Acuratețe LSTM 2')
ax.plot(thresholds, f12, marker='o', label='F1 LSTM 2')

ax.legend()
fig.suptitle('Arhitectura cu LSTM bidirecțional')
fig.savefig('lstms.png')
