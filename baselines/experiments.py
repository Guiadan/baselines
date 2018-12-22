import os
import sys

for env in ['EnduroNoFrameskip-v4', 'AmidarNoFrameskip-v4', 'BankHeistNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'AtlantisNoFrameskip-v4']:
    for seed in range(5):
        args = "-m baselines.run --alg=deepq --env={} --num_timesteps=1e6 --seed={} --neural_linear=False --ddqn=True --exp_name='ddqn'".format(seed)
        print('run #{}'.format(seed+1))
        print('python ' + args)
        os.system('/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python ' + args)
