import matplotlib.pyplot as plt
import numpy as np

#Nmin = 14
#Nmax = 17
N = np.array([29,30,22,27,23,28,25,26,24,21])#np.linspace(Nmin, Nmax, Nmax-Nmin+1).astype(int)

yticks = np.linspace(0,1.2,7)

fs = 15
bar_width = 0.35

r1 = np.arange(len(N))
# r2 = [x + bar_width for x in r1]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

## Chapel vs. CUDA

CHPL = np.array([3.82, 4.95, 5.61, 20.95, 44.04, 76.58, 89.43, 194.77, 816.01, 1455.95])
CUDA = np.array([4.18,4.91,5.63,19.82,41.04,73.75,81.97,176.40,738.93,1308.79])

axs[0].set_title("Nvidia Tesla V100", fontsize=fs)
axs[0].bar(r1[:len(CHPL)], CHPL/CUDA, color="grey", width=bar_width, zorder=2)
axs[0].axhline(y=1, color='black', linestyle='-', zorder=2)

axs[0].set_ylabel("Normalized execution time", fontsize=fs, labelpad=10)
axs[0].set_xlabel("Instance index", fontsize=fs)
axs[0].set_xticks([r for r in range(len(N))], N)
axs[0].set_yticks(yticks)
axs[0].set_ylim([0,1.2])
axs[0].tick_params(labelsize=fs)
axs[0].grid(zorder=0)

## Chapel vs. HIP

yticks = np.linspace(0,3.0,7)

CHPL = np.array([14.62, 18.85, 21.81, 81.08, 171.34, 291.99, 345.95, 733.38, 3081.55, 5589.85])
HIP  = np.array([7.56,9.14,10.52,38.08,79.44,140.81,159.35,379.45,1445.49,2538.23])

axs[1].set_title("AMD Radeon Instinct MI50", fontsize=fs)
axs[1].bar(r1[:len(CHPL)], CHPL/HIP, color="grey", width=bar_width, zorder=2, label="Chapel")
axs[1].axhline(y=1, color='black', linestyle='-', label='baseline', zorder=2)

axs[1].set_xlabel("Instance index", fontsize=fs)
axs[1].set_xticks([r for r in range(len(N))], N)
axs[1].set_yticks(yticks)
axs[1].set_ylim([0,2.5])
axs[1].tick_params(labelsize=fs)
axs[1].grid(zorder=0)

##
plt.subplots_adjust(top=0.8)
fig.legend(loc='upper center', fancybox=True, shadow=True, ncol=2, fontsize=fs)
# plt.tight_layout()
plt.savefig("singleGPU.eps")
plt.show()
