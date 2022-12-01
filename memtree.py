from distutils.log import debug
import time
import signal
import subprocess
from typing import Tuple 
import psutil
import click
import json
 
    
 
def print_memory_usage(ps, save_json):
    
 
    
    total = 0.0
    
    #total_results = []
    parts = []

    for p in ps:
        # use the "unique size set" for now:
        try:
            part = p.memory_full_info().rss
            parts.append((p.pid, part))
            total += part
        except psutil.NoSuchProcess:
            print('Process no longer exist')

    total_result = {'parts': parts,
                    'result': total}
    
            
    parts_fmt = [
        f"{pid:7}: {part/1024.0/1024.0:.2f}MiB"
        for (pid, part) in parts
    ]
    parts_fmt = "\n\t".join(parts_fmt)

    print(f"total: {total/1024.0/1024.0:.2f}MiB")
    print("\t" + parts_fmt)

    return total_result

def get_children(main_process):
    children = [
        pr for pr in psutil.process_iter() if pr.ppid() == main_process.pid
    ]
    return children


def run_command(cmd: Tuple[str]):
    p = subprocess.Popen(cmd)
    return p.pid, p


@click.command()
@click.option('-s', '--save-json', type=str)
@click.option('-p', '--pid', type=int)
@click.argument('cmd', nargs=-1, type=str)
def main(cmd, pid, save_json):
    if pid is None and not cmd:
        click.echo("either specify `pid` or `cmd`")
        return
    subp = None
    if pid is None:
        pid, subp = run_command(cmd)

    print(f"monitoring pid {pid}")
    main_process = psutil.Process(pid=pid)

    result_time_stamp = []
    
    try:
        while True:
            time_stamp = time.time()
            children = get_children(main_process)
            processes = [main_process] + children
            total_result = print_memory_usage(processes, save_json)
            time.sleep(0.2)
            total_result['time_stamp'] = time_stamp
            result_time_stamp.append(total_result)
            #with open(save_json, 'w') as f:
            #        json.dump(result_time_stamp, f)
             
            if subp is not None:
                if subp.poll() is not None:
                    break
            
    finally:
        if subp is not None:
            subp.send_signal(signal.SIGINT)
            subp.wait()


if __name__ == "__main__":
    main()
