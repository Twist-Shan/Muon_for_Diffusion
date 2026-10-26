"""Use installed system NumPy/SciPy with the existing user-site PyTorch."""
import sys
import numpy as np
import scipy.linalg
from pathlib import Path
_user_site = Path.home() / 'AppData/Roaming/Python/Python312/site-packages'
if _user_site.exists() and str(_user_site) not in sys.path:
    sys.path.append(str(_user_site))
import torch
from threadpoolctl import threadpool_limits
torch.set_num_threads(1)
_thread_limit = threadpool_limits(limits=1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
