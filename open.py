import numpy as np

data = np.load('results/original/euler_20/eval/report.npz')

print("Доступные массивы:", list(data.keys()))
print(f"BPD: {data['mean_bpd']}")
print(f"std_bpd: {data['std_bpd']}")
print(f"FID: {data['fid']}")
print(f"IS: {data['IS']}")
