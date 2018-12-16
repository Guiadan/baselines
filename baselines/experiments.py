import os

for i in range(5):
	args = "-m baselines.run --alg=deepq --env=EnduroNoFrameskip-v4 --num_timesteps=1e6 --seed={}".format(i)
	print('run #{}'.format(i+1))
	print('python ' + args)
	os.system('python ' + args)
