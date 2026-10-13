import dataclasses
import enum
import logging
import socket

import tyro
import os
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO
    name: str = "pi0_libero"
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 9999
    # Record the policy's behavior for debugging.
    record: bool = True

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    checkpoint_check_dir: str = ""

    total_task_num: int = 1
    data_name: str = "libero_spatial"
    current_task: int = 0
    exp_name : str = "BASE"
    current : int = 0





# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    # EnvMode.LIBERO: Checkpoint(
    #     config="pi05_libero",
    #     dir="/zhaolibang/openpi/checkpoints/pi05_libero/BASE/libero/libero_spatial_3_task0/5000",

    # ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_libero",
        dir="/zhaolibang/openpi/checkpoints/pi0_libero/BASE/libero/libero_object_3_task0/9999",

    ),
}


def create_default_policy(args, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(args.env):
        return _policy_config.create_trained_policy(
            _config.get_config(args.name),  args.policy.dir, default_prompt=args.default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.name), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args, default_prompt=args.default_prompt)

import pdb
def main(args: Args) -> None:


    import threading
    import socket
    import logging
    import time

    # def keepalive_listener():
    #         """永久监听 999，收到就关，保持端口存活。"""
    #         lk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #         lk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    #         lk.bind(("0.0.0.0", 999))
    #         lk.listen(5)
    #         logging.info("Keep-alive listener on :999")
    #         while True:
    #             try:
    #                 conn, addr = lk.accept()
    #                 logging.info("Keep-alive  :999")
    #                 conn.close()
    #             except Exception as e:
    #                 logging.exception("keep-alive accept error: %s", e)

    # threading.Thread(target=keepalive_listener, daemon=True).start()

    #从文件夹中读取文件名列表
    file_list = os.listdir(args.policy.dir)
    file_list = sorted(file_list,reverse=False)
    # file_list = [""]
    print(file_list)
    for file in file_list[args.current:]:
        file_splits = file.split('_')
        args.data_name = "{}".format(file_splits[0])
        args.current_task = int(file_splits[1][-1])
        args.policy.dir = args.policy.dir + "/{}_task{}".format(args.data_name,  args.current_task)
        file_lists = os.listdir(args.policy.dir)
        file_lists = [int(i) for i in file_lists]
        file_lists = sorted(file_lists, reverse=True)
        if args.current_task == 0:
            args.policy.dir = args.policy.dir + "/" + str(2999)
        elif args.exp_name == 'INFO':
            args.policy.dir = args.policy.dir + "/" + str(6000)
        else:
            args.policy.dir = args.policy.dir + "/" + str(600)

        # args.policy.dir = args.policy.dir + "/" + str(299)
        print(args.policy.dir)
        policy = create_policy(args)
        policy_metadata = policy.metadata
        
        policy_metadata['total_task_num'] = 6
        policy_metadata['data_name'] = args.data_name
        policy_metadata['current_task'] = args.current_task
        policy_metadata['name'] = args.name
        policy_metadata['exp_name'] = args.exp_name
        
        print(policy_metadata)
        # Record the policy's behavior.
        if args.record:
            policy = _policy.PolicyRecorder(policy, "policy_records")

        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        print("Creating server (host: %s, ip: %s)", hostname, local_ip)
        server = websocket_policy_server.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port, #30888, #args.port,
            metadata=policy_metadata,
        )

        # def serve():
        server.serve_forever()
        #server.serve_forever()

        # 1. 启动 server 线程
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        print(f'Server started on {9999}, will shutdown after {15000}s')

        # 2. 主线程纯睡觉
        time.sleep(15000)

        # 3. 时间到，关门
        print('Time is up, shutting down...')
        server.shutdown()        # 打断 serve_forever
        server.server_close()
        print('Server stopped.')

        




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='serve_policy.log',filemode='a',force=True)
    main(tyro.cli(Args))
