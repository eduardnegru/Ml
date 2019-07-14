import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from random import randint

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)

f1_dense1 = [
77.3149,
76.0157,
74.2851,
71.7100,
68.1406,
62.8729

]
f1_dense2 = [

77.6127,
76.0580,
73.9415,
71.1954,
67.5300,
61.9226

]
f1_dense3 = [

76.9314,
74.5790,
71.8755,
68.3051,
64.3365,
58.8414

]

f1_lstm1 = [
80.7288,
79.8716,
77.5112,
71.9997,
61.1609,
49.3825

]
f1_lstm2 = [

80.3608,
78.8740,
75.3916,
69.0853,
59.4851,
48.5937

]


thresholds = [0.4, 0.5, 0.6, 0.7,0.8, 0.9]

plt.xlabel("Prag")
plt.ylabel("Procent")
ax.plot(thresholds, f1_dense1, marker='o', label='F1 Dense 1 strat x 1024 neuorni')
ax.plot(thresholds, f1_dense2, marker='o', label='F1 Dense 2 straturi x 1024 neuroni')

ax.plot(thresholds, f1_dense3, marker='o', label='F1 Dense 3 straturi x 1024 neuroni')
ax.plot(thresholds, f1_lstm1, marker='o', label='F1 LSTM 128 neuroni')
ax.plot(thresholds, f1_lstm2, marker='o', label='F1 LSTM + dropout + maxpooling')

ax.legend()
fig.suptitle('Comparație între arhitecturi set de test Quora')
fig.savefig('lstms.png')
