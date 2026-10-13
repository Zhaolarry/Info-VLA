import os
from huggingface_hub import HfApi
from huggingface_hub import whoami
import time
# 请通过环境变量提供你的 Hugging Face token：
#   export HF_TOKEN=<your-hf-token>
HF_TOKEN = os.environ["HF_TOKEN"]
# user = whoami(token=HF_TOKEN)
api = HfApi(token=HF_TOKEN)

local_path = "/zhaolibang/datasets/libero"

#得到local_path的下一级所有文件夹名称
next_level_path = os.listdir(local_path)
print(next_level_path)
api = HfApi(token=HF_TOKEN)
for path in next_level_path:
    
    if path == 'libero_10_5_task0' or path == 'libero_10_3_task1' or path == 'libero_object_5_task0' or path == 'libero_object_5_task1' or 'libero_spatial_5' in path:
        continue
    # if api.repo_exists("libero/{}".format(path)):
    api.create_repo(repo_id="libero/{}".format(path), exist_ok=True,repo_type="dataset", private=True)
    api.upload_folder(
        folder_path=os.path.join(local_path, path), # 本地文件夹路径
   repo_id="libero/{}".format(path), # 仓库名称
   repo_type="dataset" # 仓库类型
)
    time.sleep(180)