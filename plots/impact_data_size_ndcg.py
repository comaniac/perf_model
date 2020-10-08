import argparse
import matplotlib
import matplotlib.pyplot as plt

split_ratio = [1.0, 0.7, 0.5, 0.3]
fontsize = 14
# scores = [0.8361, 0.830, 0.822, 0.734]
#
# scores_graviton2 = [0.8423, 0.8666, 0.7818, 0.6603]
# scores_rasp4b = [0.8941, 0.8710, 0.8828, 0.7999]
# scores_skylake = [0.7570, 0.7430, 0.8494, 0.7206]
# scores_t4 = [0.8207, 0.7923, 0.8195, 0.7675]
# scores_v100 = [0.8601, 0.8413, 0.8185, 0.7981]

scores = [0.8738, 0.8522, 0.8142, 0.7755]  # Large Pool

color_hex = ['#488f31', '#88a44f', '#bbba78', '#e3d2a7', '#f4c8a7', '#eb9f7d']

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)
line1, = plt.plot(split_ratio[::-1], scores[::-1], color=color_hex[0], marker='o',
                  label='Average Score')
# line2, = plt.plot(split_ratio[::-1], scores_graviton2[::-1], color=color_hex[1], marker='x',
#                   label='Graviton2')
# line3, = plt.plot(split_ratio[::-1], scores_rasp4b[::-1], color=color_hex[2], marker='.',
#                   label='Rasp4b')
# line4, = plt.plot(split_ratio[::-1], scores_skylake[::-1], color=color_hex[3], marker='o',
#                   label='Skylake')
# line5, = plt.plot(split_ratio[::-1], scores_t4[::-1], color=color_hex[4], marker='x',
#                   label='T4')
# line6, = plt.plot(split_ratio[::-1], scores_v100[::-1], color=color_hex[5], marker='x',
#                   label='V100')
font = {'size': fontsize}

matplotlib.rc('font', **font)
plt.xticks(split_ratio[::-1], size=fontsize)
plt.yticks(size=fontsize)
plt.xlim([0.22, 1.05])
plt.ylim([0.75, 0.89])
plt.legend(handles=[line1], loc='lower right')
plt.xlabel('Sample Ratio', size=fontsize)
plt.ylabel('NDCG@8', size=fontsize)
for i, v in zip(split_ratio[::-1], scores[::-1]):
    if i == 1.0:
        ax.annotate(str(v), xy=(i - 0.078, v + 0.008))
    elif i == 0.5:
        ax.annotate(str(v), xy=(i - 0.078, v + 0.010))
    else:
        ax.annotate(str(v), xy=(i - 0.058, v + 0.012))
fig.savefig('impact_data_size.pdf', bbox_inches='tight')
fig.savefig('impact_data_size.png', bbox_inches='tight')
