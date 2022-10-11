#!python3

import os
import sys
import time
import datetime
import math
import random
import tqdm
import glob
import psutil
import argparse

import socket
import subprocess
from multiprocessing import Pool as ProcessPool

import torch


def test_gpu(args):
    print('[test]', vars(args))
    D = 1024
    vram_bytes_i = D * D * 4
    vram_bytes = float(args.hold) * (1024 ** 3)
    N = max(int(vram_bytes / vram_bytes_i * 0.77), 3)
    print(D, 'x', D, N)
    m_list = [torch.randn(D, D).type(torch.float32).cuda() for _ in range(0, N)]
    with torch.no_grad():
        while True:
            for i in range(0, N):
                m_list[i] = torch.matmul(m_list[i], m_list[i])
                m_list[i] -= m_list[i].mean()
                m_list[i] /= m_list[i].std()
            time.sleep(0.25)


def cmd_executor(cmd_list):
    t0 = time.time()
    for i in range(0, len(cmd_list)):
        assert len(cmd_list[i]) == 3, str(cmd_list[i])
        c, e, o = cmd_list[i]
        # time.sleep(2)
        with open(o, 'w') as fp:
            p = subprocess.Popen(c, env=e, stdout=fp, stderr=fp)
            p.wait()
        print('[%d/%d finished]' % (i + 1, len(cmd_list)), '[%.1f hours]' % ((time.time() - t0) / 3600.0), '[%s]' % ' '.join(c), '>>>', '[%s]' % o, flush=True)


def run_adapt(args):
    # python train_net_intersections.py --id 001
    basedir = os.path.dirname(__file__)
    assert os.access(os.path.join(basedir, 'train_net_intersections.py'), os.R_OK)
    assert len(args.gpus) > 0
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt_intersections_%s_lr0.00010_iter20000.pth' % x), os.R_OK), args.ids))
    random.shuffle(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()

    commands = []
    for i in range(0, len(args.gpus)):
        commands_i = []
        if i < len(args.gpus) - 1:
            vids_batch = vids[len(vids) // len(args.gpus) * i : len(vids) // len(args.gpus) * (i + 1)]
        else:
            vids_batch = vids[len(vids) // len(args.gpus) * i :]
        vids_batch = sorted(vids_batch)
        env_i = curr_env.copy()
        env_i['CUDA_VISIBLE_DEVICES'] = str(args.gpus[i])
        for v in vids_batch:
            log_i = 'log_adapt_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'train_net_intersections.py'),  '--id', v]
            commands_i.append([cmd_i, env_i, log_i])
        commands_i.append([[python_path, os.path.join(basedir, 'run_videos.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_adapt_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)

    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


def run_adapt_ddp_2gpus(args):
    # python train_net_intersections.py --id 001 --ddp_num_gpus 2
    basedir = os.path.dirname(__file__)
    assert os.access(os.path.join(basedir, 'train_net_intersections.py'), os.R_OK)
    assert len(args.gpus) > 0
    vids = list(filter(lambda x: not os.access(os.path.join(basedir, 'adapt_intersections_%s_lr0.00010_iter20000.pth' % x), os.R_OK), args.ids))
    vids = sorted(vids)
    print(vids)

    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
    assert len(args.gpus) == 2, 'only supports 2-GPUs DDP'
    commands = []
    curr_env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, args.gpus))
    print('CUDA_VISIBLE_DEVICES=%s' % curr_env['CUDA_VISIBLE_DEVICES'], flush=True)

    for v in vids:
        commands.append(
            (
                [python_path, os.path.join(basedir, 'train_net_intersections.py'),  '--id', v, '--ddp_num_gpus', '2', '--ddp_port', args.ddp_port],
                curr_env,
                os.path.join(basedir, 'log_adapt_%s_%s_GPU%s.log' % (v, socket.gethostname(), curr_env['CUDA_VISIBLE_DEVICES']))
            )
        )
    commands.append(
        (
            [python_path, os.path.join(basedir, 'run_videos.py'), '--opt', 'test', '--hold', args.hold],
            curr_env,
            os.path.join(basedir, 'log_adapt_999_%s_GPU%s.log' % (socket.gethostname(), curr_env['CUDA_VISIBLE_DEVICES']))
        )
    )
    cmd_executor(commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str)
    parser.add_argument('--ids', nargs='+', default=[])
    parser.add_argument('--gpus', nargs='+', default=[])
    parser.add_argument('--hold', type=str, default='0.005')
    parser.add_argument('--ddp_port', type=str, default='50152')
    args = parser.parse_args()
    print(args)

    if args.opt == 'adapt':
        run_adapt(args)
    if args.opt == 'adapt_ddp2':
        run_adapt_ddp_2gpus(args)
    elif args.opt == 'test':
        test_gpu(args)
    else: pass
    exit(0)

'''
nohup python run_videos.py --opt adapt --gpus 0 1 2 --hold 9.27 --ids 001 003 005 006 007 008 009 011 012 013 014 015 016 017 019 020 023 025 027 034 036 039 040 043 044 046 048 049 050 051 053 054 055 056 058 059 060 066 067 068 069 070 071 073 074 075 076 077 080 085 086 087 088 090 091 092 093 094 095 098 099 105 108 110 112 114 115 116 117 118 125 127 128 129 130 131 132 135 136 141 146 148 149 150 152 154 156 158 159 160 161 164 167 169 170 171 172 175 178 179 &

nohup python run_videos.py --opt adapt_ddp2 --gpus 0 1 --hold 9.27 --ids 001 003 005 006 007 008 009 011 012 013 014 015 016 017 019 020 023 025 027 034 036 039 040 043 044 046 048 049 050 051 053 054 055 056 058 059 060 066 067 068 069 070 071 073 074 075 076 077 080 085 086 087 088 090 091 092 093 094 095 098 099 105 108 110 112 114 115 116 117 118 125 127 128 129 130 131 132 135 136 141 146 148 149 150 152 154 156 158 159 160 161 164 167 169 170 171 172 175 178 179 &
'''
