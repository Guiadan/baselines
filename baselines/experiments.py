import os
import sys
from datetime import datetime

fmt = '%Y-%m-%d %H:%M:%S'
ddqn =[False, False, True, True]
exp_name = ['nl_no_prior', 'nl_simple_prior', 'nl_no_prior_ddqn', 'nl_simple_ddqn']
prior = ['no prior', 'simple', 'no prior', 'simple']
for i in range(1):
    for env in ['EnduroNoFrameskip-v4', 'AmidarNoFrameskip-v4', 'BankHeistNoFrameskip-v4', 'AsterixNoFrameskip-v4', 'AtlantisNoFrameskip-v4']:
        for seed in range(5):
            args = "-m baselines.run --alg=deepq --env={} --num_timesteps=1e7 --seed={} --neural_linear=True --ddqn={}" \
                   " --exp_name={} --prior={}".format(env, seed, ddqn[i], exp_name[i],prior[i])
            print('run #{}'.format(seed+1))
            print('python ' + args)
            t1 = datetime.now()
            os.system('/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python ' + args)
            t2 = datetime.now()
            hour_diff = abs(t2.hour - t1.hour)
            if hour_diff < 1:
                min_diff = abs(t2.minute - t1.minute)
                if min_diff < 15:
                    exit(0)
