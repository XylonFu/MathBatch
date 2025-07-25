# models/api_server.py
import os
import signal
import subprocess
import time

import psutil
import torch

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


def start_vllm_server(conda_env_path, model_path, served_model_name,
                      devices=None, tensor_parallel_size=4, max_model_len=16384, max_num_seqs=512,
                      host="127.0.0.1", port=8000, api_key="EMPTY", chat_template=None):
    if devices is None:
        devices = [0, 1, 2, 3]
    devices_str = ",".join(str(d) for d in devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str
    cmd = [
        "conda", "run", "--prefix", os.path.expandvars(conda_env_path), "--no-capture-output",
        "vllm", "serve", model_path,
        "--served-model-name", served_model_name,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(max_num_seqs),
        "--host", host,
        "--port", str(port),
        "--api-key", api_key,
    ]
    if chat_template is not None:
        cmd.extend(["--chat-template", chat_template])
    process = subprocess.Popen(cmd, env=env, start_new_session=True)
    return process


def start_lmdeploy_server(conda_env_path, model_path, served_model_name,
                          devices=None, tensor_parallel_size=4, max_model_len=16384, max_num_seqs=512,
                          host="127.0.0.1", port=8000, api_key="EMPTY", chat_template=None):
    if devices is None:
        devices = [0, 1, 2, 3]
    devices_str = ",".join(str(d) for d in devices)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = devices_str
    cmd = [
        "conda", "run", "--prefix", os.path.expandvars(conda_env_path), "--no-capture-output",
        "lmdeploy", "serve", "api_server", model_path,
        "--model-name", served_model_name,
        "--tp", str(tensor_parallel_size),
        "--session-len", str(max_model_len),
        "--max-batch-size", str(max_num_seqs),
        "--server-name", host,
        "--server-port", str(port),
        "--api-keys", api_key,
    ]
    if chat_template is not None:
        cmd.extend(["--chat-template", chat_template])
    process = subprocess.Popen(cmd, env=env, start_new_session=True)
    return process


def wait_server(host="127.0.0.1", port=8000, timeout=600):
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            import requests
            response = requests.get(url)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"Server didn't become ready within {timeout}s")


def query_gpu_pids(device_id):
    pids = set()
    if NVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in procs:
                pids.add(p.pid)
            pynvml.nvmlShutdown()
            return list(pids)
        except Exception:
            pass

    try:
        result = subprocess.run(
            f"nvidia-smi --query-compute-apps=pid --format=csv,noheader -i {device_id}",
            shell=True, capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line.split()[0])
                pids.add(pid)
            except:
                continue
    except Exception:
        pass
    return list(pids)


def kill_pid_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except Exception:
            pass
    try:
        parent.kill()
    except Exception:
        pass


def wait_gpu_memory_released(device_id, timeout=60, poll_interval=5):
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                f"nvidia-smi --query-gpu=memory.used --format=csv,noheader -i {device_id}",
                shell=True, capture_output=True, text=True, check=True
            )
            used_str = result.stdout.strip().split()[0]
            used_mem = int(used_str)
        except Exception:
            used_mem = None

        if used_mem is None:
            time.sleep(poll_interval)
            continue

        if used_mem <= 10:
            return True

        time.sleep(poll_interval)
    return False


def stop_server(process, devices=None, wait_timeout=60):
    pgid = None
    try:
        pgid = os.getpgid(process.pid)
        os.killpg(pgid, signal.SIGTERM)
    except Exception:
        try:
            process.terminate()
        except Exception:
            pass

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.wait()

    if devices is not None:
        device_list = [devices] if isinstance(devices, int) else list(devices)
        for device_id in device_list:
            pids = query_gpu_pids(device_id)
            for pid in pids:
                try:
                    p = psutil.Process(pid)
                    cmdline = " ".join(p.cmdline())
                    if "vllm.worker" in cmdline or "vllm.serve" in cmdline or "vllm serve" in cmdline:
                        kill_pid_tree(pid)
                except Exception:
                    pass
            ok = wait_gpu_memory_released(device_id, timeout=wait_timeout)
            if not ok:
                print(f"[Warning] GPU {device_id} memory not fully released after timeout.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
