"""Self-scheduling task worker for short contam_ tasks (runs inside any Slurm job).
Tasks: code/iqueue/tasks.txt, one per line:  name|prereqs(comma)|done_file|est_min|command
A task runs when all prereq files exist, its done_file does not, and no LIVE job holds its lock.
Locks whose job is no longer in squeue are stale and are taken over. Up to C tasks in parallel.
Any number of workers (interactive or normal partitions) can share the queue safely."""
import os, sys, time, subprocess, signal
W = '.'; Q = f'{W}/code/iqueue'; L = f'{Q}/locks'; os.makedirs(L, exist_ok=True)
JOB = os.environ.get('SLURM_JOB_ID', 'local'); C = int(os.environ.get('IW_CONC', '4'))
T0 = time.time()
def tlimit_min():
    try:
        out = subprocess.run(['squeue', '-h', '-j', JOB, '-o', '%L'], capture_output=True, text=True, timeout=60).stdout.strip()
        parts = [int(x) for x in out.replace('-', ':').split(':')]
        while len(parts) < 3: parts.insert(0, 0)
        d = parts[0] * 0 if len(parts) == 3 else parts[0]
        h, m, s = parts[-3:]; return d * 1440 + h * 60 + m + s / 60
    except Exception: return 50.0
END = T0 + 60 * (tlimit_min() - 3)
def live(j):
    if j == JOB: return True
    try: return bool(subprocess.run(['squeue', '-h', '-j', j], capture_output=True, text=True, timeout=60).stdout.strip())
    except Exception: return True
def ex(p): return os.path.exists(p if p.startswith('/') else f'{W}/{p}')
def tasks():
    out = []
    for ln in open(os.environ.get('IW_TASKS', f'{Q}/tasks.txt')):
        ln = ln.strip()
        if not ln or ln.startswith('#'): continue
        n, pre, done, est, cmd = ln.split('|', 4)
        out.append((n, [p for p in pre.split(',') if p], done, float(est), cmd))
    return out
def claim(n):
    lp = f'{L}/{n}.lock'
    for _ in range(2):
        try:
            fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, JOB.encode()); os.close(fd); return True
        except FileExistsError:
            try: holder = open(lp).read().strip()
            except Exception: return False
            if live(holder): return False
            try: os.remove(lp)
            except Exception: return False
    return False
running = {}
failed_here = set()
print(f'worker {JOB} start, budget {(END-T0)/60:.0f} min, conc {C}', flush=True)
while True:
    for n, (p, lp) in list(running.items()):
        if p.poll() is not None:
            print(f'[{time.strftime("%H:%M:%S")}] END {n} rc={p.returncode}', flush=True)
            if p.returncode != 0: failed_here.add(n)
            try: os.remove(lp)
            except Exception: pass
            del running[n]
    left = (END - time.time()) / 60
    todo = [t for t in tasks() if not ex(t[2]) and t[0] not in failed_here]
    if not todo and not running: print('queue empty', flush=True); break
    if left < 2:
        for n, (p, lp) in running.items():
            print('time up, stopping', n, flush=True); os.killpg(p.pid, signal.SIGTERM)
        time.sleep(10); break
    if len(running) < C:
        for n, pre, done, est, cmd in todo:
            if n in running or n in failed_here or len(running) >= C: continue
            if est > left or not all(ex(x) for x in pre): continue
            if not claim(n): continue
            log = open(f'{W}/logs/iq_{n}_{JOB}.log', 'w')
            p = subprocess.Popen(['bash', '-lc', f'cd {W} && source env.sh >/dev/null 2>&1; export OMP_NUM_THREADS=2; {cmd}'],
                                 stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
            running[n] = (p, f'{L}/{n}.lock')
            print(f'[{time.strftime("%H:%M:%S")}] START {n} (left {left:.0f} min)', flush=True)
    time.sleep(20)
print(f'worker {JOB} done', flush=True)
