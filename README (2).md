```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
pip install ultralytics
```

    Collecting ultralytics
      Downloading ultralytics-8.3.161-py3-none-any.whl.metadata (37 kB)
    Requirement already satisfied: numpy>=1.23.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (2.0.2)
    Requirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (3.10.0)
    Requirement already satisfied: opencv-python>=4.6.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (4.11.0.86)
    Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (11.2.1)
    Requirement already satisfied: pyyaml>=5.3.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (6.0.2)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (2.32.3)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (1.15.3)
    Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (2.6.0+cu124)
    Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (0.21.0+cu124)
    Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (4.67.1)
    Requirement already satisfied: psutil in /usr/local/lib/python3.11/dist-packages (from ultralytics) (5.9.5)
    Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.11/dist-packages (from ultralytics) (9.0.0)
    Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.11/dist-packages (from ultralytics) (2.2.2)
    Collecting ultralytics-thop>=2.0.0 (from ultralytics)
      Downloading ultralytics_thop-2.0.14-py3-none-any.whl.metadata (9.4 kB)
    Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (4.58.4)
    Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (1.4.8)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (24.2)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.11/dist-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.1.4->ultralytics) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas>=1.1.4->ultralytics) (2025.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics) (2.4.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests>=2.23.0->ultralytics) (2025.6.15)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (3.18.0)
    Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (4.14.0)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (3.5)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (3.1.6)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (2025.3.2)
    Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cublas-cu12==12.4.5.8 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cufft-cu12==11.2.1.3 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-curand-cu12==10.3.5.147 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Requirement already satisfied: nvidia-cusparselt-cu12==0.6.2 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (0.6.2)
    Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (12.4.127)
    Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch>=1.8.0->ultralytics)
      Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Requirement already satisfied: triton==3.2.0 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (3.2.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch>=1.8.0->ultralytics) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch>=1.8.0->ultralytics) (1.3.0)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.17.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch>=1.8.0->ultralytics) (3.0.2)
    Downloading ultralytics-8.3.161-py3-none-any.whl (1.0 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m29.0 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m363.4/363.4 MB[0m [31m4.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.8/13.8 MB[0m [31m125.3 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m24.6/24.6 MB[0m [31m103.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m883.7/883.7 kB[0m [31m60.5 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m664.8/664.8 MB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.5/211.5 MB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.3/56.3 MB[0m [31m13.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m127.9/127.9 MB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m207.5/207.5 MB[0m [31m5.9 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m43.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading ultralytics_thop-2.0.14-py3-none-any.whl (26 kB)
    Installing collected packages: nvidia-nvjitlink-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, ultralytics-thop, ultralytics
      Attempting uninstall: nvidia-nvjitlink-cu12
        Found existing installation: nvidia-nvjitlink-cu12 12.5.82
        Uninstalling nvidia-nvjitlink-cu12-12.5.82:
          Successfully uninstalled nvidia-nvjitlink-cu12-12.5.82
      Attempting uninstall: nvidia-curand-cu12
        Found existing installation: nvidia-curand-cu12 10.3.6.82
        Uninstalling nvidia-curand-cu12-10.3.6.82:
          Successfully uninstalled nvidia-curand-cu12-10.3.6.82
      Attempting uninstall: nvidia-cufft-cu12
        Found existing installation: nvidia-cufft-cu12 11.2.3.61
        Uninstalling nvidia-cufft-cu12-11.2.3.61:
          Successfully uninstalled nvidia-cufft-cu12-11.2.3.61
      Attempting uninstall: nvidia-cuda-runtime-cu12
        Found existing installation: nvidia-cuda-runtime-cu12 12.5.82
        Uninstalling nvidia-cuda-runtime-cu12-12.5.82:
          Successfully uninstalled nvidia-cuda-runtime-cu12-12.5.82
      Attempting uninstall: nvidia-cuda-nvrtc-cu12
        Found existing installation: nvidia-cuda-nvrtc-cu12 12.5.82
        Uninstalling nvidia-cuda-nvrtc-cu12-12.5.82:
          Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.5.82
      Attempting uninstall: nvidia-cuda-cupti-cu12
        Found existing installation: nvidia-cuda-cupti-cu12 12.5.82
        Uninstalling nvidia-cuda-cupti-cu12-12.5.82:
          Successfully uninstalled nvidia-cuda-cupti-cu12-12.5.82
      Attempting uninstall: nvidia-cublas-cu12
        Found existing installation: nvidia-cublas-cu12 12.5.3.2
        Uninstalling nvidia-cublas-cu12-12.5.3.2:
          Successfully uninstalled nvidia-cublas-cu12-12.5.3.2
      Attempting uninstall: nvidia-cusparse-cu12
        Found existing installation: nvidia-cusparse-cu12 12.5.1.3
        Uninstalling nvidia-cusparse-cu12-12.5.1.3:
          Successfully uninstalled nvidia-cusparse-cu12-12.5.1.3
      Attempting uninstall: nvidia-cudnn-cu12
        Found existing installation: nvidia-cudnn-cu12 9.3.0.75
        Uninstalling nvidia-cudnn-cu12-9.3.0.75:
          Successfully uninstalled nvidia-cudnn-cu12-9.3.0.75
      Attempting uninstall: nvidia-cusolver-cu12
        Found existing installation: nvidia-cusolver-cu12 11.6.3.83
        Uninstalling nvidia-cusolver-cu12-11.6.3.83:
          Successfully uninstalled nvidia-cusolver-cu12-11.6.3.83
    Successfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127 ultralytics-8.3.161 ultralytics-thop-2.0.14



```python
#tracking with Byte_track
#results=model.track(source='/content/drive/MyDrive/15sec_input_720p.mp4',show=True,save=True,tracker='bytetrack.yaml',conf=0.20,iou=0.3)
```

    
    WARNING ⚠️ 
    inference results will accumulate in RAM unless `stream=True` is passed, causing potential out-of-memory
    errors for large sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.
    
    Example:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
    
    video 1/1 (frame 1/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 153.2ms
    video 1/1 (frame 2/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 138.3ms
    video 1/1 (frame 3/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 173.1ms
    video 1/1 (frame 4/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 121.3ms
    video 1/1 (frame 5/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 132.1ms
    video 1/1 (frame 6/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 133.3ms
    video 1/1 (frame 7/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 127.1ms
    video 1/1 (frame 8/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 127.0ms
    video 1/1 (frame 9/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 128.8ms
    video 1/1 (frame 10/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 130.4ms
    video 1/1 (frame 11/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 136.6ms
    video 1/1 (frame 12/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 137.7ms
    video 1/1 (frame 13/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 156.5ms
    video 1/1 (frame 14/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 138.8ms
    video 1/1 (frame 15/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 133.3ms
    video 1/1 (frame 16/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 139.5ms
    video 1/1 (frame 17/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 143.2ms
    video 1/1 (frame 18/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 130.1ms
    video 1/1 (frame 19/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 128.9ms
    video 1/1 (frame 20/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 130.5ms
    video 1/1 (frame 21/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 145.6ms
    video 1/1 (frame 22/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 131.5ms
    video 1/1 (frame 23/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 137.7ms
    video 1/1 (frame 24/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 130.5ms
    video 1/1 (frame 25/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 133.2ms
    video 1/1 (frame 26/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 129.0ms
    video 1/1 (frame 27/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 126.6ms
    video 1/1 (frame 28/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 129.8ms
    video 1/1 (frame 29/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 152.3ms
    video 1/1 (frame 30/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 201.9ms
    video 1/1 (frame 31/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 216.3ms
    video 1/1 (frame 32/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 208.8ms
    video 1/1 (frame 33/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 187.2ms
    video 1/1 (frame 34/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 185.0ms
    video 1/1 (frame 35/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 238.6ms
    video 1/1 (frame 36/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 210.9ms
    video 1/1 (frame 37/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 189.1ms
    video 1/1 (frame 38/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 190.2ms
    video 1/1 (frame 39/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 210.9ms
    video 1/1 (frame 40/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 199.0ms
    video 1/1 (frame 41/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 205.3ms
    video 1/1 (frame 42/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 242.7ms
    video 1/1 (frame 43/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 210.7ms
    video 1/1 (frame 44/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 222.1ms
    video 1/1 (frame 45/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 216.8ms
    video 1/1 (frame 46/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 134.3ms
    video 1/1 (frame 47/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 151.0ms
    video 1/1 (frame 48/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 122.2ms
    video 1/1 (frame 49/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 140.0ms
    video 1/1 (frame 50/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 135.4ms
    video 1/1 (frame 51/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 133.7ms
    video 1/1 (frame 52/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 142.3ms
    video 1/1 (frame 53/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 137.4ms
    video 1/1 (frame 54/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 134.1ms
    video 1/1 (frame 55/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 139.5ms
    video 1/1 (frame 56/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 135.8ms
    video 1/1 (frame 57/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 133.0ms
    video 1/1 (frame 58/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 131.2ms
    video 1/1 (frame 59/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 138.5ms
    video 1/1 (frame 60/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 135.7ms
    video 1/1 (frame 61/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 163.0ms
    video 1/1 (frame 62/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 127.8ms
    video 1/1 (frame 63/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 146.5ms
    video 1/1 (frame 64/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 131.9ms
    video 1/1 (frame 65/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 129.2ms
    video 1/1 (frame 66/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 128.5ms
    video 1/1 (frame 67/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 147.1ms
    video 1/1 (frame 68/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 133.8ms
    video 1/1 (frame 69/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 146.6ms
    video 1/1 (frame 70/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 136.1ms
    video 1/1 (frame 71/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 130.5ms
    video 1/1 (frame 72/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 125.8ms
    video 1/1 (frame 73/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 139.9ms
    video 1/1 (frame 74/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 145.5ms
    video 1/1 (frame 75/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 133.8ms
    video 1/1 (frame 76/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 129.3ms
    video 1/1 (frame 77/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 139.6ms
    video 1/1 (frame 78/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 1 sports ball, 153.1ms
    video 1/1 (frame 79/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 149.6ms
    video 1/1 (frame 80/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 160.3ms
    video 1/1 (frame 81/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 138.1ms
    video 1/1 (frame 82/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 129.0ms
    video 1/1 (frame 83/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 126.1ms
    video 1/1 (frame 84/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 144.0ms
    video 1/1 (frame 85/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 140.3ms
    video 1/1 (frame 86/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 139.2ms
    video 1/1 (frame 87/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 128.7ms
    video 1/1 (frame 88/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 124.9ms
    video 1/1 (frame 89/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 130.7ms
    video 1/1 (frame 90/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 134.6ms
    video 1/1 (frame 91/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 144.6ms
    video 1/1 (frame 92/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 132.4ms
    video 1/1 (frame 93/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 137.5ms
    video 1/1 (frame 94/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 132.9ms
    video 1/1 (frame 95/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 135.4ms
    video 1/1 (frame 96/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 15 persons, 147.7ms
    video 1/1 (frame 97/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 148.9ms
    video 1/1 (frame 98/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 140.0ms
    video 1/1 (frame 99/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 127.2ms
    video 1/1 (frame 100/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 141.8ms
    video 1/1 (frame 101/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 127.4ms
    video 1/1 (frame 102/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 133.9ms
    video 1/1 (frame 103/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 247.3ms
    video 1/1 (frame 104/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 201.0ms
    video 1/1 (frame 105/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 17 persons, 192.8ms
    video 1/1 (frame 106/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 16 persons, 194.4ms
    video 1/1 (frame 107/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 215.2ms
    video 1/1 (frame 108/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 201.9ms
    video 1/1 (frame 109/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 249.0ms
    video 1/1 (frame 110/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 187.1ms
    video 1/1 (frame 111/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 210.6ms
    video 1/1 (frame 112/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 193.3ms
    video 1/1 (frame 113/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 221.6ms
    video 1/1 (frame 114/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 194.1ms
    video 1/1 (frame 115/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 217.2ms
    video 1/1 (frame 116/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 203.6ms
    video 1/1 (frame 117/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 237.7ms
    video 1/1 (frame 118/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 207.3ms
    video 1/1 (frame 119/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 145.4ms
    video 1/1 (frame 120/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 142.4ms
    video 1/1 (frame 121/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 129.2ms
    video 1/1 (frame 122/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 144.5ms
    video 1/1 (frame 123/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 124.1ms
    video 1/1 (frame 124/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 139.6ms
    video 1/1 (frame 125/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 128.1ms
    video 1/1 (frame 126/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 141.7ms
    video 1/1 (frame 127/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 132.6ms
    video 1/1 (frame 128/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 126.6ms
    video 1/1 (frame 129/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 132.9ms
    video 1/1 (frame 130/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 150.1ms
    video 1/1 (frame 131/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 131.9ms
    video 1/1 (frame 132/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 153.5ms
    video 1/1 (frame 133/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 138.1ms
    video 1/1 (frame 134/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 138.4ms
    video 1/1 (frame 135/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 147.7ms
    video 1/1 (frame 136/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 131.8ms
    video 1/1 (frame 137/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 131.8ms
    video 1/1 (frame 138/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 138.3ms
    video 1/1 (frame 139/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 119.1ms
    video 1/1 (frame 140/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 127.4ms
    video 1/1 (frame 141/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 125.3ms
    video 1/1 (frame 142/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 144.5ms
    video 1/1 (frame 143/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 123.0ms
    video 1/1 (frame 144/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 142.3ms
    video 1/1 (frame 145/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 133.6ms
    video 1/1 (frame 146/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 123.1ms
    video 1/1 (frame 147/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 1 sports ball, 128.5ms
    video 1/1 (frame 148/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 154.8ms
    video 1/1 (frame 149/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 128.7ms
    video 1/1 (frame 150/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 153.5ms
    video 1/1 (frame 151/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 137.4ms
    video 1/1 (frame 152/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 129.5ms
    video 1/1 (frame 153/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 132.7ms
    video 1/1 (frame 154/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 14 persons, 134.9ms
    video 1/1 (frame 155/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 133.0ms
    video 1/1 (frame 156/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 153.9ms
    video 1/1 (frame 157/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 126.9ms
    video 1/1 (frame 158/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 134.3ms
    video 1/1 (frame 159/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 145.2ms
    video 1/1 (frame 160/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 128.8ms
    video 1/1 (frame 161/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 127.7ms
    video 1/1 (frame 162/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 130.0ms
    video 1/1 (frame 163/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 131.4ms
    video 1/1 (frame 164/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 127.6ms
    video 1/1 (frame 165/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 138.9ms
    video 1/1 (frame 166/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 128.3ms
    video 1/1 (frame 167/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 126.8ms
    video 1/1 (frame 168/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 136.3ms
    video 1/1 (frame 169/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 164.6ms
    video 1/1 (frame 170/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 143.6ms
    video 1/1 (frame 171/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 134.9ms
    video 1/1 (frame 172/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 144.4ms
    video 1/1 (frame 173/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 131.9ms
    video 1/1 (frame 174/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 138.8ms
    video 1/1 (frame 175/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 148.4ms
    video 1/1 (frame 176/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 132.7ms
    video 1/1 (frame 177/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 13 persons, 188.7ms
    video 1/1 (frame 178/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 198.8ms
    video 1/1 (frame 179/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 238.1ms
    video 1/1 (frame 180/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 251.3ms
    video 1/1 (frame 181/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 197.2ms
    video 1/1 (frame 182/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 199.4ms
    video 1/1 (frame 183/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 195.4ms
    video 1/1 (frame 184/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 209.2ms
    video 1/1 (frame 185/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 200.5ms
    video 1/1 (frame 186/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 201.8ms
    video 1/1 (frame 187/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 227.0ms
    video 1/1 (frame 188/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 200.7ms
    video 1/1 (frame 189/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 195.6ms
    video 1/1 (frame 190/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 233.5ms
    video 1/1 (frame 191/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 208.7ms
    video 1/1 (frame 192/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 202.5ms
    video 1/1 (frame 193/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 122.4ms
    video 1/1 (frame 194/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 127.1ms
    video 1/1 (frame 195/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 130.6ms
    video 1/1 (frame 196/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 133.5ms
    video 1/1 (frame 197/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 130.9ms
    video 1/1 (frame 198/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 139.8ms
    video 1/1 (frame 199/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 125.1ms
    video 1/1 (frame 200/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 129.5ms
    video 1/1 (frame 201/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 153.0ms
    video 1/1 (frame 202/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 132.8ms
    video 1/1 (frame 203/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 146.3ms
    video 1/1 (frame 204/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 152.1ms
    video 1/1 (frame 205/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 132.2ms
    video 1/1 (frame 206/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 134.9ms
    video 1/1 (frame 207/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 157.6ms
    video 1/1 (frame 208/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 1 sports ball, 135.6ms
    video 1/1 (frame 209/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 1 sports ball, 149.5ms
    video 1/1 (frame 210/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 1 sports ball, 151.5ms
    video 1/1 (frame 211/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 1 sports ball, 127.7ms
    video 1/1 (frame 212/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 1 sports ball, 139.0ms
    video 1/1 (frame 213/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 1 sports ball, 130.2ms
    video 1/1 (frame 214/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 1 sports ball, 157.3ms
    video 1/1 (frame 215/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 1 sports ball, 129.7ms
    video 1/1 (frame 216/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 1 sports ball, 154.0ms
    video 1/1 (frame 217/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 1 sports ball, 131.6ms
    video 1/1 (frame 218/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 130.4ms
    video 1/1 (frame 219/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 124.6ms
    video 1/1 (frame 220/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 130.2ms
    video 1/1 (frame 221/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 138.0ms
    video 1/1 (frame 222/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 155.2ms
    video 1/1 (frame 223/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 141.8ms
    video 1/1 (frame 224/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 127.5ms
    video 1/1 (frame 225/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 132.8ms
    video 1/1 (frame 226/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 138.5ms
    video 1/1 (frame 227/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 131.4ms
    video 1/1 (frame 228/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 143.1ms
    video 1/1 (frame 229/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 137.7ms
    video 1/1 (frame 230/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 149.6ms
    video 1/1 (frame 231/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 131.4ms
    video 1/1 (frame 232/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 132.1ms
    video 1/1 (frame 233/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 132.2ms
    video 1/1 (frame 234/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 131.5ms
    video 1/1 (frame 235/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 132.0ms
    video 1/1 (frame 236/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 129.5ms
    video 1/1 (frame 237/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 142.3ms
    video 1/1 (frame 238/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 153.8ms
    video 1/1 (frame 239/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 128.0ms
    video 1/1 (frame 240/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 122.6ms
    video 1/1 (frame 241/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 148.2ms
    video 1/1 (frame 242/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 131.8ms
    video 1/1 (frame 243/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 135.2ms
    video 1/1 (frame 244/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 151.0ms
    video 1/1 (frame 245/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 132.3ms
    video 1/1 (frame 246/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 133.2ms
    video 1/1 (frame 247/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 154.8ms
    video 1/1 (frame 248/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 151.8ms
    video 1/1 (frame 249/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 154.0ms
    video 1/1 (frame 250/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 129.2ms
    video 1/1 (frame 251/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 156.1ms
    video 1/1 (frame 252/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 212.4ms
    video 1/1 (frame 253/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 190.9ms
    video 1/1 (frame 254/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 188.8ms
    video 1/1 (frame 255/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 209.9ms
    video 1/1 (frame 256/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 203.4ms
    video 1/1 (frame 257/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 254.1ms
    video 1/1 (frame 258/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 203.4ms
    video 1/1 (frame 259/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 201.0ms
    video 1/1 (frame 260/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 201.4ms
    video 1/1 (frame 261/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 241.7ms
    video 1/1 (frame 262/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 205.4ms
    video 1/1 (frame 263/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 206.2ms
    video 1/1 (frame 264/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 204.6ms
    video 1/1 (frame 265/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 220.0ms
    video 1/1 (frame 266/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 244.1ms
    video 1/1 (frame 267/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 204.2ms
    video 1/1 (frame 268/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 147.8ms
    video 1/1 (frame 269/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 151.9ms
    video 1/1 (frame 270/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 136.3ms
    video 1/1 (frame 271/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 143.2ms
    video 1/1 (frame 272/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 129.1ms
    video 1/1 (frame 273/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 123.8ms
    video 1/1 (frame 274/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 135.0ms
    video 1/1 (frame 275/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 131.2ms
    video 1/1 (frame 276/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 176.9ms
    video 1/1 (frame 277/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 132.0ms
    video 1/1 (frame 278/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 139.3ms
    video 1/1 (frame 279/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 131.2ms
    video 1/1 (frame 280/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 129.4ms
    video 1/1 (frame 281/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 5 persons, 129.0ms
    video 1/1 (frame 282/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 153.0ms
    video 1/1 (frame 283/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 138.0ms
    video 1/1 (frame 284/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 139.6ms
    video 1/1 (frame 285/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 138.3ms
    video 1/1 (frame 286/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 159.2ms
    video 1/1 (frame 287/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 131.3ms
    video 1/1 (frame 288/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 170.8ms
    video 1/1 (frame 289/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 148.4ms
    video 1/1 (frame 290/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 133.8ms
    video 1/1 (frame 291/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 133.6ms
    video 1/1 (frame 292/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 137.5ms
    video 1/1 (frame 293/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 147.9ms
    video 1/1 (frame 294/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 132.5ms
    video 1/1 (frame 295/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 126.2ms
    video 1/1 (frame 296/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 130.8ms
    video 1/1 (frame 297/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 140.3ms
    video 1/1 (frame 298/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 131.6ms
    video 1/1 (frame 299/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 138.0ms
    video 1/1 (frame 300/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 151.1ms
    video 1/1 (frame 301/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 135.3ms
    video 1/1 (frame 302/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 138.2ms
    video 1/1 (frame 303/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 140.6ms
    video 1/1 (frame 304/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 127.0ms
    video 1/1 (frame 305/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 143.2ms
    video 1/1 (frame 306/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 141.2ms
    video 1/1 (frame 307/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 135.6ms
    video 1/1 (frame 308/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 136.6ms
    video 1/1 (frame 309/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 12 persons, 141.3ms
    video 1/1 (frame 310/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 123.6ms
    video 1/1 (frame 311/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 139.7ms
    video 1/1 (frame 312/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 154.5ms
    video 1/1 (frame 313/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 137.0ms
    video 1/1 (frame 314/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 134.2ms
    video 1/1 (frame 315/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 133.2ms
    video 1/1 (frame 316/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 133.9ms
    video 1/1 (frame 317/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 140.4ms
    video 1/1 (frame 318/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 150.6ms
    video 1/1 (frame 319/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 138.2ms
    video 1/1 (frame 320/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 11 persons, 147.6ms
    video 1/1 (frame 321/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 138.1ms
    video 1/1 (frame 322/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 135.1ms
    video 1/1 (frame 323/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 141.8ms
    video 1/1 (frame 324/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 10 persons, 143.2ms
    video 1/1 (frame 325/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 139.7ms
    video 1/1 (frame 326/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 209.7ms
    video 1/1 (frame 327/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 255.4ms
    video 1/1 (frame 328/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 200.4ms
    video 1/1 (frame 329/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 192.7ms
    video 1/1 (frame 330/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 205.7ms
    video 1/1 (frame 331/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 213.8ms
    video 1/1 (frame 332/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 215.2ms
    video 1/1 (frame 333/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 241.8ms
    video 1/1 (frame 334/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 195.3ms
    video 1/1 (frame 335/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 235.4ms
    video 1/1 (frame 336/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 198.8ms
    video 1/1 (frame 337/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 210.5ms
    video 1/1 (frame 338/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 218.3ms
    video 1/1 (frame 339/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 248.6ms
    video 1/1 (frame 340/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 209.7ms
    video 1/1 (frame 341/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 221.3ms
    video 1/1 (frame 342/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 143.0ms
    video 1/1 (frame 343/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 146.3ms
    video 1/1 (frame 344/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 135.5ms
    video 1/1 (frame 345/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 132.6ms
    video 1/1 (frame 346/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 160.4ms
    video 1/1 (frame 347/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 140.6ms
    video 1/1 (frame 348/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 141.4ms
    video 1/1 (frame 349/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 6 persons, 124.7ms
    video 1/1 (frame 350/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 127.1ms
    video 1/1 (frame 351/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 126.9ms
    video 1/1 (frame 352/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 152.8ms
    video 1/1 (frame 353/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 144.0ms
    video 1/1 (frame 354/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 159.3ms
    video 1/1 (frame 355/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 140.5ms
    video 1/1 (frame 356/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 133.2ms
    video 1/1 (frame 357/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 129.5ms
    video 1/1 (frame 358/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 141.4ms
    video 1/1 (frame 359/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 132.8ms
    video 1/1 (frame 360/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 123.2ms
    video 1/1 (frame 361/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 9 persons, 128.3ms
    video 1/1 (frame 362/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 138.6ms
    video 1/1 (frame 363/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 130.3ms
    video 1/1 (frame 364/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 164.1ms
    video 1/1 (frame 365/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 7 persons, 131.3ms
    video 1/1 (frame 366/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 128.5ms
    video 1/1 (frame 367/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 146.4ms
    video 1/1 (frame 368/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 137.9ms
    video 1/1 (frame 369/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 154.8ms
    video 1/1 (frame 370/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 153.3ms
    video 1/1 (frame 371/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 8 persons, 151.9ms
    video 1/1 (frame 372/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 3 persons, 1 baseball glove, 131.9ms
    video 1/1 (frame 373/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 2 persons, 138.9ms
    video 1/1 (frame 374/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 2 persons, 137.7ms
    video 1/1 (frame 375/375) /content/drive/MyDrive/15sec_input_720p.mp4: 384x640 2 persons, 129.2ms
    Speed: 2.9ms preprocess, 153.5ms inference, 1.6ms postprocess per image at shape (1, 3, 384, 640)
    Results saved to [1mruns/detect/track2[0m



```python
#python code using OpenCV and YOLO11 to run Object Trcaking on Video Frames and on Live Webcam feed

import cv2
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

#Load the YOLO11 Model
model=YOLO('yolo11n.pt')

cap=cv2.VideoCapture('/content/drive/MyDrive/15sec_input_720p.mp4')

# Process a fixed number of frames (e.g., 100 frames)
max_frames = 100
frame_count = 0

while True:
  ret,frame=cap.read()
  if ret and frame_count < max_frames:
    result=model.track(frame,persist=True)
    annotated_frame=result[0].plot()
    cv2_imshow(annotated_frame)

    frame_count += 1
  else:
    break
cap.release()
results=model.track(source='/content/drive/MyDrive/15sec_input_720p.mp4',show=True,save=True)
```

    
    0: 384x640 17 persons, 322.9ms
    Speed: 19.4ms preprocess, 322.9ms inference, 34.7ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_1.png)
    


    
    0: 384x640 16 persons, 209.6ms
    Speed: 3.2ms preprocess, 209.6ms inference, 2.2ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_3.png)
    


    
    0: 384x640 15 persons, 241.7ms
    Speed: 3.0ms preprocess, 241.7ms inference, 2.2ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_5.png)
    


    
    0: 384x640 13 persons, 235.9ms
    Speed: 3.4ms preprocess, 235.9ms inference, 2.1ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_7.png)
    


    
    0: 384x640 14 persons, 240.9ms
    Speed: 3.1ms preprocess, 240.9ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_9.png)
    


    
    0: 384x640 15 persons, 215.8ms
    Speed: 10.3ms preprocess, 215.8ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_11.png)
    


    
    0: 384x640 15 persons, 268.7ms
    Speed: 8.2ms preprocess, 268.7ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_13.png)
    


    
    0: 384x640 15 persons, 265.0ms
    Speed: 3.2ms preprocess, 265.0ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_15.png)
    


    
    0: 384x640 15 persons, 252.0ms
    Speed: 4.7ms preprocess, 252.0ms inference, 2.4ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_17.png)
    


    
    0: 384x640 14 persons, 216.8ms
    Speed: 3.1ms preprocess, 216.8ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_19.png)
    


    
    0: 384x640 15 persons, 239.9ms
    Speed: 3.7ms preprocess, 239.9ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_21.png)
    


    
    0: 384x640 14 persons, 223.0ms
    Speed: 2.9ms preprocess, 223.0ms inference, 1.7ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_23.png)
    


    
    0: 384x640 14 persons, 254.2ms
    Speed: 4.9ms preprocess, 254.2ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_25.png)
    


    
    0: 384x640 13 persons, 220.8ms
    Speed: 3.3ms preprocess, 220.8ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_27.png)
    


    
    0: 384x640 13 persons, 260.9ms
    Speed: 5.1ms preprocess, 260.9ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_29.png)
    


    
    0: 384x640 14 persons, 285.0ms
    Speed: 3.2ms preprocess, 285.0ms inference, 2.1ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_31.png)
    


    
    0: 384x640 14 persons, 307.6ms
    Speed: 3.2ms preprocess, 307.6ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_33.png)
    


    
    0: 384x640 14 persons, 242.3ms
    Speed: 3.8ms preprocess, 242.3ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_35.png)
    


    
    0: 384x640 14 persons, 231.5ms
    Speed: 4.7ms preprocess, 231.5ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_37.png)
    


    
    0: 384x640 15 persons, 215.6ms
    Speed: 3.2ms preprocess, 215.6ms inference, 3.3ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_39.png)
    


    
    0: 384x640 14 persons, 245.2ms
    Speed: 3.2ms preprocess, 245.2ms inference, 3.2ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_41.png)
    


    
    0: 384x640 16 persons, 223.2ms
    Speed: 3.3ms preprocess, 223.2ms inference, 2.1ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_43.png)
    


    
    0: 384x640 16 persons, 207.3ms
    Speed: 6.0ms preprocess, 207.3ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_45.png)
    


    
    0: 384x640 17 persons, 222.3ms
    Speed: 3.1ms preprocess, 222.3ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_47.png)
    


    
    0: 384x640 17 persons, 234.4ms
    Speed: 4.5ms preprocess, 234.4ms inference, 2.5ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_49.png)
    


    
    0: 384x640 17 persons, 202.8ms
    Speed: 3.1ms preprocess, 202.8ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_51.png)
    


    
    0: 384x640 16 persons, 217.1ms
    Speed: 3.7ms preprocess, 217.1ms inference, 2.1ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_53.png)
    


    
    0: 384x640 17 persons, 270.0ms
    Speed: 3.4ms preprocess, 270.0ms inference, 2.4ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_55.png)
    


    
    0: 384x640 17 persons, 243.1ms
    Speed: 3.4ms preprocess, 243.1ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_57.png)
    


    
    0: 384x640 15 persons, 245.2ms
    Speed: 3.2ms preprocess, 245.2ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_59.png)
    


    
    0: 384x640 14 persons, 213.6ms
    Speed: 6.1ms preprocess, 213.6ms inference, 1.7ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_61.png)
    


    
    0: 384x640 15 persons, 258.8ms
    Speed: 4.3ms preprocess, 258.8ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_63.png)
    


    
    0: 384x640 15 persons, 223.5ms
    Speed: 3.5ms preprocess, 223.5ms inference, 2.3ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_65.png)
    


    
    0: 384x640 14 persons, 244.6ms
    Speed: 3.3ms preprocess, 244.6ms inference, 1.7ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_67.png)
    


    
    0: 384x640 14 persons, 239.5ms
    Speed: 3.8ms preprocess, 239.5ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_69.png)
    


    
    0: 384x640 14 persons, 246.6ms
    Speed: 5.1ms preprocess, 246.6ms inference, 2.4ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_71.png)
    


    
    0: 384x640 15 persons, 243.5ms
    Speed: 3.2ms preprocess, 243.5ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_73.png)
    


    
    0: 384x640 16 persons, 295.6ms
    Speed: 11.4ms preprocess, 295.6ms inference, 2.1ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_75.png)
    


    
    0: 384x640 16 persons, 257.3ms
    Speed: 3.7ms preprocess, 257.3ms inference, 2.0ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_77.png)
    


    
    0: 384x640 16 persons, 236.1ms
    Speed: 2.9ms preprocess, 236.1ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_79.png)
    


    
    0: 384x640 16 persons, 235.1ms
    Speed: 3.3ms preprocess, 235.1ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_81.png)
    


    
    0: 384x640 17 persons, 278.7ms
    Speed: 3.4ms preprocess, 278.7ms inference, 1.9ms postprocess per image at shape (1, 3, 384, 640)



    
![png](output_3_83.png)
    


    
    0: 384x640 17 persons, 263.7ms
    Speed: 3.2ms preprocess, 263.7ms inference, 2.4ms postprocess per image at shape (1, 3, 384, 640)
    Buffered data was truncated after reaching the output size limit.


```python
import json
import time
from google.colab import files
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('/content/drive/MyDrive/best.pt')
video_path = '/content/drive/MyDrive/15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video_path = '/content/output_l.mp4'
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
output_data = []

# Initialize ReID feature storage (simple centroid and size)
reid_features = {}  # {id: (centroid_x, centroid_y, width, height)}
reid_threshold = 25  # Distance threshold for ReID matching

def extract_features(bbox):
    x1, y1, x2, y2 = bbox
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return (centroid_x, centroid_y, width, height)

def reidentify(prev_id, new_id, prev_bbox, new_bbox):
    prev_feat = extract_features(prev_bbox)
    new_feat = extract_features(new_bbox)
    dist = np.sqrt((prev_feat[0] - new_feat[0])**2 + (prev_feat[1] - new_feat[1])**2)
    size_diff = abs(prev_feat[2] * prev_feat[3] - new_feat[2] * new_feat[3])
    if dist < reid_threshold and size_diff < 100:
        return prev_id
    return new_id

# Process video in real-time simulation
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    results = model.track(frame, conf=0.6, iou=0.5, persist=True)
    frame_data = {'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)), 'players': []}

    if results[0].boxes.id is not None:
        current_ids = results[0].boxes.id.int().tolist()
        current_boxes = results[0].boxes.xyxy.int().tolist()

        # ReID and merge logic
        for box, id in zip(current_boxes, current_ids):
            x1, y1, x2, y2 = box
            new_feature = extract_features(box)
            matched = False

            for prev_id, prev_feat in reid_features.items():
                if reidentify(prev_id, id, prev_feat[1], box) == prev_id:
                    id = prev_id
                    matched = True
                    break

            if not matched:
                reid_features[id] = (frame_count, new_feature)
            frame_data['players'].append({'id': int(id), 'bbox': [x1, y1, x2, y2]})

    output_data.append(frame_data)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # Simulate real-time delay
    elapsed = time.time() - start_time
    if elapsed < 1.0 / fps:
        time.sleep(1.0 / fps - elapsed)

    frame_count += 1
    print(f"Processed frame {frame_count}")

cap.release()
out.release()

# Load existing JSON for merging (if available), otherwise use new data
try:
    with open('/content/player_tracking_merged (2).json', 'r') as f:
        merged_data = json.load(f)
    merged_data.extend(output_data)
    output_data = merged_data
except FileNotFoundError:
    pass

# Update ID merging logic
merged_ids = {}
for i in range(len(output_data) - 1):
    frame1 = output_data[i]['players']
    frame2 = output_data[i + 1]['players']
    for p1 in frame1:
        for p2 in frame2:
            if p1['id'] != p2['id']:
                bbox1 = p1['bbox']
                bbox2 = p2['bbox']
                x1, y1, x2, y2 = bbox1
                x3, y3, x4, y4 = bbox2
                w1, h1 = x2 - x1, y2 - y1
                w2, h2 = x4 - x3, y4 - y3
                if (abs(x1 - x3) < 20 and abs(y1 - y3) < 20 and
                    max(w1, w2) - min(w1, w2) < 10 and max(h1, h2) - min(h1, h2) < 10):
                    merged_ids[p2['id']] = p1['id'] if w1 * h1 > w2 * h2 else p2['id']

# Remove duplicates within the same frame
for frame in output_data:
    seen_ids = {}
    for player in frame['players']:
        id = player['id']
        bbox = player['bbox']
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if id in seen_ids:
            if area > seen_ids[id]['area']:
                frame['players'].remove(seen_ids[id]['player'])
            else:
                frame['players'].remove(player)
                continue
        seen_ids[id] = {'player': player, 'area': area}

# Update IDs in data based on merged_ids
for frame in output_data:
    for player in frame['players']:
        if player['id'] in merged_ids:
            player['id'] = merged_ids[player['id']]

# Save the updated JSON file
with open('/content/player_tracking_merged_updated.json', 'w') as f:
    json.dump(output_data, f, indent=2)

# Download the file
files.download('/content/player_tracking_merged_updated.json')
```

    
    0: 384x640 16 players, 1 referee, 1922.3ms
    Speed: 2.0ms preprocess, 1922.3ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 1
    
    0: 384x640 15 players, 2 referees, 1926.2ms
    Speed: 1.8ms preprocess, 1926.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 2
    
    0: 384x640 15 players, 2 referees, 1913.4ms
    Speed: 2.2ms preprocess, 1913.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 3
    
    0: 384x640 14 players, 1 referee, 1929.9ms
    Speed: 1.9ms preprocess, 1929.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 4
    
    0: 384x640 14 players, 2 referees, 2303.5ms
    Speed: 1.7ms preprocess, 2303.5ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 5
    
    0: 384x640 14 players, 2 referees, 1974.2ms
    Speed: 1.7ms preprocess, 1974.2ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 6
    
    0: 384x640 15 players, 1 referee, 1974.3ms
    Speed: 1.8ms preprocess, 1974.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 7
    
    0: 384x640 15 players, 1 referee, 1941.2ms
    Speed: 1.8ms preprocess, 1941.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 8
    
    0: 384x640 15 players, 1 referee, 1942.6ms
    Speed: 1.7ms preprocess, 1942.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 9
    
    0: 384x640 15 players, 1 referee, 1940.1ms
    Speed: 2.6ms preprocess, 1940.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 10
    
    0: 384x640 14 players, 2 referees, 2383.7ms
    Speed: 2.0ms preprocess, 2383.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 11
    
    0: 384x640 14 players, 1 referee, 1928.2ms
    Speed: 1.8ms preprocess, 1928.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 12
    
    0: 384x640 13 players, 2 referees, 1928.8ms
    Speed: 1.9ms preprocess, 1928.8ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 13
    
    0: 384x640 13 players, 2 referees, 1923.1ms
    Speed: 2.0ms preprocess, 1923.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 14
    
    0: 384x640 13 players, 2 referees, 1908.7ms
    Speed: 2.8ms preprocess, 1908.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 15
    
    0: 384x640 12 players, 2 referees, 1933.7ms
    Speed: 2.0ms preprocess, 1933.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 16
    
    0: 384x640 12 players, 2 referees, 2421.2ms
    Speed: 1.8ms preprocess, 2421.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 17
    
    0: 384x640 12 players, 2 referees, 1917.1ms
    Speed: 1.8ms preprocess, 1917.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 18
    
    0: 384x640 13 players, 2 referees, 1923.8ms
    Speed: 1.8ms preprocess, 1923.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 19
    
    0: 384x640 13 players, 2 referees, 1916.4ms
    Speed: 1.5ms preprocess, 1916.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 20
    
    0: 384x640 14 players, 1 referee, 1943.1ms
    Speed: 2.0ms preprocess, 1943.1ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 21
    
    0: 384x640 13 players, 2 referees, 2025.9ms
    Speed: 1.8ms preprocess, 2025.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 22
    
    0: 384x640 14 players, 2 referees, 2332.0ms
    Speed: 1.7ms preprocess, 2332.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 23
    
    0: 384x640 14 players, 2 referees, 1963.5ms
    Speed: 1.8ms preprocess, 1963.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 24
    
    0: 384x640 14 players, 2 referees, 1908.9ms
    Speed: 1.7ms preprocess, 1908.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 25
    
    0: 384x640 13 players, 2 referees, 1915.7ms
    Speed: 1.8ms preprocess, 1915.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 26
    
    0: 384x640 13 players, 2 referees, 1929.0ms
    Speed: 1.7ms preprocess, 1929.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 27
    
    0: 384x640 13 players, 2 referees, 2035.8ms
    Speed: 1.8ms preprocess, 2035.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 28
    
    0: 384x640 13 players, 2 referees, 2438.8ms
    Speed: 2.2ms preprocess, 2438.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 29
    
    0: 384x640 13 players, 2 referees, 1936.6ms
    Speed: 1.5ms preprocess, 1936.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 30
    
    0: 384x640 13 players, 2 referees, 1923.5ms
    Speed: 3.0ms preprocess, 1923.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 31
    
    0: 384x640 12 players, 2 referees, 1933.9ms
    Speed: 1.9ms preprocess, 1933.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 32
    
    0: 384x640 12 players, 2 referees, 1914.2ms
    Speed: 2.7ms preprocess, 1914.2ms inference, 10.5ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 33
    
    0: 384x640 12 players, 2 referees, 2165.8ms
    Speed: 2.5ms preprocess, 2165.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 34
    
    0: 384x640 12 players, 2 referees, 2087.7ms
    Speed: 2.0ms preprocess, 2087.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 35
    
    0: 384x640 13 players, 2 referees, 1952.0ms
    Speed: 1.9ms preprocess, 1952.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 36
    
    0: 384x640 13 players, 2 referees, 1932.5ms
    Speed: 2.9ms preprocess, 1932.5ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 37
    
    0: 384x640 14 players, 2 referees, 1918.0ms
    Speed: 1.6ms preprocess, 1918.0ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 38
    
    0: 384x640 14 players, 2 referees, 1936.5ms
    Speed: 2.7ms preprocess, 1936.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 39
    
    0: 384x640 16 players, 3 referees, 2232.0ms
    Speed: 2.0ms preprocess, 2232.0ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 40
    
    0: 384x640 16 players, 2 referees, 2092.4ms
    Speed: 1.7ms preprocess, 2092.4ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 41
    
    0: 384x640 15 players, 2 referees, 1950.4ms
    Speed: 2.2ms preprocess, 1950.4ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 42
    
    0: 384x640 14 players, 2 referees, 1929.5ms
    Speed: 1.9ms preprocess, 1929.5ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 43
    
    0: 384x640 15 players, 2 referees, 1941.8ms
    Speed: 2.9ms preprocess, 1941.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 44
    
    0: 384x640 14 players, 2 referees, 1898.3ms
    Speed: 1.8ms preprocess, 1898.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 45
    
    0: 384x640 15 players, 2 referees, 2304.8ms
    Speed: 2.5ms preprocess, 2304.8ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 46
    
    0: 384x640 14 players, 2 referees, 2058.7ms
    Speed: 1.9ms preprocess, 2058.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 47
    
    0: 384x640 15 players, 2 referees, 1938.0ms
    Speed: 1.9ms preprocess, 1938.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 48
    
    0: 384x640 15 players, 2 referees, 1925.6ms
    Speed: 1.9ms preprocess, 1925.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 49
    
    0: 384x640 15 players, 2 referees, 1923.3ms
    Speed: 3.1ms preprocess, 1923.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 50
    
    0: 384x640 16 players, 2 referees, 1941.6ms
    Speed: 1.8ms preprocess, 1941.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 51
    
    0: 384x640 16 players, 1 referee, 2362.9ms
    Speed: 1.8ms preprocess, 2362.9ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 52
    
    0: 384x640 14 players, 3 referees, 1906.9ms
    Speed: 2.2ms preprocess, 1906.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 53
    
    0: 384x640 14 players, 3 referees, 1916.8ms
    Speed: 2.4ms preprocess, 1916.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 54
    
    0: 384x640 13 players, 3 referees, 1928.6ms
    Speed: 2.6ms preprocess, 1928.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 55
    
    0: 384x640 15 players, 2 referees, 1901.9ms
    Speed: 1.9ms preprocess, 1901.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 56
    
    0: 384x640 13 players, 2 referees, 1913.6ms
    Speed: 1.9ms preprocess, 1913.6ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 57
    
    0: 384x640 13 players, 2 referees, 2425.6ms
    Speed: 2.0ms preprocess, 2425.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 58
    
    0: 384x640 14 players, 2 referees, 1921.9ms
    Speed: 1.9ms preprocess, 1921.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 59
    
    0: 384x640 14 players, 2 referees, 1903.9ms
    Speed: 3.6ms preprocess, 1903.9ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 60
    
    0: 384x640 14 players, 2 referees, 1945.2ms
    Speed: 3.6ms preprocess, 1945.2ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 61
    
    0: 384x640 15 players, 1 referee, 1921.6ms
    Speed: 1.7ms preprocess, 1921.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 62
    
    0: 384x640 14 players, 2 referees, 1917.8ms
    Speed: 1.9ms preprocess, 1917.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 63
    
    0: 384x640 14 players, 2 referees, 2375.7ms
    Speed: 4.1ms preprocess, 2375.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 64
    
    0: 384x640 14 players, 2 referees, 1924.7ms
    Speed: 1.7ms preprocess, 1924.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 65
    
    0: 384x640 13 players, 4 referees, 1910.9ms
    Speed: 1.9ms preprocess, 1910.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 66
    
    0: 384x640 14 players, 3 referees, 1903.0ms
    Speed: 1.9ms preprocess, 1903.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 67
    
    0: 384x640 13 players, 3 referees, 1923.8ms
    Speed: 3.7ms preprocess, 1923.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 68
    
    0: 384x640 14 players, 2 referees, 1943.6ms
    Speed: 2.4ms preprocess, 1943.6ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 69
    
    0: 384x640 14 players, 2 referees, 2318.4ms
    Speed: 1.7ms preprocess, 2318.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 70
    
    0: 384x640 14 players, 2 referees, 1907.8ms
    Speed: 1.9ms preprocess, 1907.8ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 71
    
    0: 384x640 14 players, 2 referees, 1904.4ms
    Speed: 1.8ms preprocess, 1904.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 72
    
    0: 384x640 14 players, 2 referees, 1907.8ms
    Speed: 2.2ms preprocess, 1907.8ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 73
    
    0: 384x640 15 players, 1 referee, 1973.6ms
    Speed: 3.3ms preprocess, 1973.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 74
    
    0: 384x640 14 players, 2 referees, 2019.1ms
    Speed: 2.3ms preprocess, 2019.1ms inference, 2.6ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 75
    
    0: 384x640 15 players, 1 referee, 2271.8ms
    Speed: 1.8ms preprocess, 2271.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 76
    
    0: 384x640 15 players, 2 referees, 1903.1ms
    Speed: 1.9ms preprocess, 1903.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 77
    
    0: 384x640 14 players, 1 referee, 1926.1ms
    Speed: 2.2ms preprocess, 1926.1ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 78
    
    0: 384x640 15 players, 2 referees, 1912.9ms
    Speed: 2.6ms preprocess, 1912.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 79
    
    0: 384x640 14 players, 2 referees, 1920.0ms
    Speed: 2.0ms preprocess, 1920.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 80
    
    0: 384x640 14 players, 2 referees, 2045.2ms
    Speed: 1.9ms preprocess, 2045.2ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 81
    
    0: 384x640 14 players, 2 referees, 2227.9ms
    Speed: 2.0ms preprocess, 2227.9ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 82
    
    0: 384x640 14 players, 2 referees, 1893.9ms
    Speed: 3.0ms preprocess, 1893.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 83
    
    0: 384x640 14 players, 2 referees, 1912.3ms
    Speed: 2.0ms preprocess, 1912.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 84
    
    0: 384x640 14 players, 1 referee, 1919.8ms
    Speed: 2.2ms preprocess, 1919.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 85
    
    0: 384x640 14 players, 2 referees, 1900.4ms
    Speed: 1.9ms preprocess, 1900.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 86
    
    0: 384x640 14 players, 2 referees, 2051.9ms
    Speed: 2.4ms preprocess, 2051.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 87
    
    0: 384x640 14 players, 1 referee, 2231.1ms
    Speed: 1.7ms preprocess, 2231.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 88
    
    0: 384x640 14 players, 1 referee, 1905.6ms
    Speed: 1.8ms preprocess, 1905.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 89
    
    0: 384x640 15 players, 1 referee, 1905.0ms
    Speed: 3.5ms preprocess, 1905.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 90
    
    0: 384x640 16 players, 1 referee, 1992.4ms
    Speed: 2.0ms preprocess, 1992.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 91
    
    0: 384x640 15 players, 2 referees, 1933.1ms
    Speed: 1.9ms preprocess, 1933.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 92
    
    0: 384x640 14 players, 2 referees, 2192.7ms
    Speed: 1.8ms preprocess, 2192.7ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 93
    
    0: 384x640 14 players, 2 referees, 2105.8ms
    Speed: 2.5ms preprocess, 2105.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 94
    
    0: 384x640 14 players, 2 referees, 1903.8ms
    Speed: 1.7ms preprocess, 1903.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 95
    
    0: 384x640 15 players, 2 referees, 1940.9ms
    Speed: 1.8ms preprocess, 1940.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 96
    
    0: 384x640 15 players, 2 referees, 1937.1ms
    Speed: 2.4ms preprocess, 1937.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 97
    
    0: 384x640 16 players, 1 referee, 2152.1ms
    Speed: 2.0ms preprocess, 2152.1ms inference, 2.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 98
    
    0: 384x640 15 players, 2 referees, 2511.4ms
    Speed: 3.2ms preprocess, 2511.4ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 99
    
    0: 384x640 15 players, 2 referees, 1927.5ms
    Speed: 2.0ms preprocess, 1927.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 100
    
    0: 384x640 15 players, 2 referees, 1917.1ms
    Speed: 2.9ms preprocess, 1917.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 101
    
    0: 384x640 17 players, 1904.9ms
    Speed: 1.8ms preprocess, 1904.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 102
    
    0: 384x640 16 players, 1 referee, 1913.7ms
    Speed: 1.9ms preprocess, 1913.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 103
    
    0: 384x640 16 players, 2 referees, 1912.3ms
    Speed: 1.9ms preprocess, 1912.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 104
    
    0: 384x640 16 players, 1 referee, 2409.2ms
    Speed: 2.0ms preprocess, 2409.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 105
    
    0: 384x640 16 players, 2 referees, 1928.2ms
    Speed: 2.1ms preprocess, 1928.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 106
    
    0: 384x640 16 players, 2 referees, 1904.6ms
    Speed: 1.7ms preprocess, 1904.6ms inference, 1.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 107
    
    0: 384x640 15 players, 1 referee, 1948.6ms
    Speed: 1.9ms preprocess, 1948.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 108
    
    0: 384x640 16 players, 1960.0ms
    Speed: 1.7ms preprocess, 1960.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 109
    
    0: 384x640 16 players, 1916.5ms
    Speed: 2.4ms preprocess, 1916.5ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 110
    
    0: 384x640 15 players, 2356.5ms
    Speed: 1.8ms preprocess, 2356.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 111
    
    0: 384x640 14 players, 1 referee, 1927.3ms
    Speed: 1.7ms preprocess, 1927.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 112
    
    0: 384x640 14 players, 1 referee, 1918.5ms
    Speed: 2.5ms preprocess, 1918.5ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 113
    
    0: 384x640 13 players, 2 referees, 1903.1ms
    Speed: 3.2ms preprocess, 1903.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 114
    
    0: 384x640 14 players, 1 referee, 1916.0ms
    Speed: 1.9ms preprocess, 1916.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 115
    
    0: 384x640 15 players, 1 referee, 1931.0ms
    Speed: 2.1ms preprocess, 1931.0ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 116
    
    0: 384x640 14 players, 1 referee, 2337.2ms
    Speed: 1.8ms preprocess, 2337.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 117
    
    0: 384x640 14 players, 1 referee, 1933.0ms
    Speed: 1.8ms preprocess, 1933.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 118
    
    0: 384x640 14 players, 1 referee, 1914.3ms
    Speed: 1.9ms preprocess, 1914.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 119
    
    0: 384x640 13 players, 1 referee, 1931.0ms
    Speed: 1.9ms preprocess, 1931.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 120
    
    0: 384x640 13 players, 1 referee, 1904.1ms
    Speed: 1.9ms preprocess, 1904.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 121
    
    0: 384x640 13 players, 1 referee, 1983.1ms
    Speed: 3.2ms preprocess, 1983.1ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 122
    
    0: 384x640 13 players, 1 referee, 2279.1ms
    Speed: 1.9ms preprocess, 2279.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 123
    
    0: 384x640 14 players, 1 referee, 1922.8ms
    Speed: 2.0ms preprocess, 1922.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 124
    
    0: 384x640 14 players, 1 referee, 1961.3ms
    Speed: 2.7ms preprocess, 1961.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 125
    
    0: 384x640 14 players, 1 referee, 1932.1ms
    Speed: 2.0ms preprocess, 1932.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 126
    
    0: 384x640 14 players, 1 referee, 1937.3ms
    Speed: 2.1ms preprocess, 1937.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 127
    
    0: 384x640 14 players, 1 referee, 2034.1ms
    Speed: 3.2ms preprocess, 2034.1ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 128
    
    0: 384x640 14 players, 1 referee, 2219.7ms
    Speed: 2.6ms preprocess, 2219.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 129
    
    0: 384x640 13 players, 1 referee, 1942.1ms
    Speed: 1.9ms preprocess, 1942.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 130
    
    0: 384x640 13 players, 1 referee, 1911.8ms
    Speed: 1.9ms preprocess, 1911.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 131
    
    0: 384x640 13 players, 1 referee, 1915.2ms
    Speed: 1.7ms preprocess, 1915.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 132
    
    0: 384x640 12 players, 1 referee, 1914.9ms
    Speed: 2.1ms preprocess, 1914.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 133
    
    0: 384x640 12 players, 1 referee, 2106.3ms
    Speed: 2.0ms preprocess, 2106.3ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 134
    
    0: 384x640 13 players, 1 referee, 2142.4ms
    Speed: 1.8ms preprocess, 2142.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 135
    
    0: 384x640 13 players, 1 referee, 1907.1ms
    Speed: 1.9ms preprocess, 1907.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 136
    
    0: 384x640 14 players, 1 referee, 1896.2ms
    Speed: 1.8ms preprocess, 1896.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 137
    
    0: 384x640 14 players, 1 referee, 1927.6ms
    Speed: 2.1ms preprocess, 1927.6ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 138
    
    0: 384x640 13 players, 1 referee, 1936.7ms
    Speed: 1.9ms preprocess, 1936.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 139
    
    0: 384x640 13 players, 1 referee, 2135.3ms
    Speed: 1.8ms preprocess, 2135.3ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 140
    
    0: 384x640 13 players, 1 referee, 2178.4ms
    Speed: 2.1ms preprocess, 2178.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 141
    
    0: 384x640 13 players, 1 referee, 1914.0ms
    Speed: 2.3ms preprocess, 1914.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 142
    
    0: 384x640 13 players, 1929.7ms
    Speed: 1.8ms preprocess, 1929.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 143
    
    0: 384x640 13 players, 1 referee, 1886.8ms
    Speed: 1.8ms preprocess, 1886.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 144
    
    0: 384x640 12 players, 1 referee, 1918.8ms
    Speed: 1.7ms preprocess, 1918.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 145
    
    0: 384x640 12 players, 1 referee, 2170.1ms
    Speed: 1.7ms preprocess, 2170.1ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 146
    
    0: 384x640 12 players, 1 referee, 2107.3ms
    Speed: 2.1ms preprocess, 2107.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 147
    
    0: 384x640 13 players, 1 referee, 1954.0ms
    Speed: 2.6ms preprocess, 1954.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 148
    
    0: 384x640 13 players, 1 referee, 1913.3ms
    Speed: 1.8ms preprocess, 1913.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 149
    
    Speed: 1.8ms preprocess, 1919.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 150
    
    0: 384x640 13 players, 1 referee, 1921.1ms
    Speed: 3.0ms preprocess, 1921.1ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 151
    
    0: 384x640 13 players, 1 referee, 2199.9ms
    Speed: 1.8ms preprocess, 2199.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 152
    
    0: 384x640 13 players, 1 referee, 2089.6ms
    Speed: 1.8ms preprocess, 2089.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 153
    
    0: 384x640 13 players, 1 referee, 1907.1ms
    Speed: 1.8ms preprocess, 1907.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 154
    
    0: 384x640 13 players, 1 referee, 1923.9ms
    Speed: 2.9ms preprocess, 1923.9ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 155
    
    0: 384x640 13 players, 1 referee, 1939.4ms
    Speed: 2.1ms preprocess, 1939.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 156
    
    0: 384x640 12 players, 1 referee, 1936.6ms
    Speed: 1.9ms preprocess, 1936.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 157
    
    0: 384x640 12 players, 1 referee, 2243.4ms
    Speed: 1.8ms preprocess, 2243.4ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 158
    
    0: 384x640 12 players, 1 referee, 2000.1ms
    Speed: 1.7ms preprocess, 2000.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 159
    
    0: 384x640 12 players, 1 referee, 1938.7ms
    Speed: 2.2ms preprocess, 1938.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 160
    
    0: 384x640 12 players, 1 referee, 1902.4ms
    Speed: 1.7ms preprocess, 1902.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 161
    
    0: 384x640 12 players, 1 referee, 1898.4ms
    Speed: 1.9ms preprocess, 1898.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 162
    
    0: 384x640 12 players, 1 referee, 1932.3ms
    Speed: 2.4ms preprocess, 1932.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 163
    
    0: 384x640 12 players, 1 referee, 2328.6ms
    Speed: 1.9ms preprocess, 2328.6ms inference, 1.4ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 164
    
    0: 384x640 12 players, 1 referee, 1925.8ms
    Speed: 2.6ms preprocess, 1925.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 165
    
    0: 384x640 12 players, 1 referee, 1892.9ms
    Speed: 2.8ms preprocess, 1892.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 166
    
    0: 384x640 12 players, 1 referee, 1908.3ms
    Speed: 2.2ms preprocess, 1908.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 167
    
    0: 384x640 13 players, 1925.8ms
    Speed: 1.8ms preprocess, 1925.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 168
    
    0: 384x640 13 players, 1924.9ms
    Speed: 1.7ms preprocess, 1924.9ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 169
    
    0: 384x640 13 players, 2366.5ms
    Speed: 2.1ms preprocess, 2366.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 170
    
    0: 384x640 12 players, 1904.1ms
    Speed: 2.5ms preprocess, 1904.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 171
    
    0: 384x640 12 players, 1 referee, 1917.4ms
    Speed: 1.8ms preprocess, 1917.4ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 172
    
    0: 384x640 12 players, 1 referee, 1971.9ms
    Speed: 1.8ms preprocess, 1971.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 173
    
    0: 384x640 13 players, 1 referee, 1940.0ms
    Speed: 2.6ms preprocess, 1940.0ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 174
    
    0: 384x640 13 players, 1953.7ms
    Speed: 2.0ms preprocess, 1953.7ms inference, 1.4ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 175
    
    0: 384x640 13 players, 2344.8ms
    Speed: 4.2ms preprocess, 2344.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 176
    
    0: 384x640 13 players, 1903.2ms
    Speed: 1.8ms preprocess, 1903.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 177
    
    0: 384x640 13 players, 1909.6ms
    Speed: 2.3ms preprocess, 1909.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 178
    
    0: 384x640 12 players, 1 referee, 1946.2ms
    Speed: 3.0ms preprocess, 1946.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 179
    
    0: 384x640 11 players, 1 referee, 1920.4ms
    Speed: 2.8ms preprocess, 1920.4ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 180
    
    0: 384x640 11 players, 1 referee, 1998.1ms
    Speed: 3.4ms preprocess, 1998.1ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 181
    
    0: 384x640 11 players, 1 referee, 2272.4ms
    Speed: 1.8ms preprocess, 2272.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 182
    
    0: 384x640 12 players, 1921.4ms
    Speed: 1.8ms preprocess, 1921.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 183
    
    0: 384x640 11 players, 1 referee, 1923.6ms
    Speed: 1.9ms preprocess, 1923.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 184
    
    0: 384x640 12 players, 1923.1ms
    Speed: 1.9ms preprocess, 1923.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 185
    
    0: 384x640 12 players, 1940.1ms
    Speed: 2.7ms preprocess, 1940.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 186
    
    0: 384x640 12 players, 2110.2ms
    Speed: 2.1ms preprocess, 2110.2ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 187
    
    0: 384x640 11 players, 1 referee, 2155.0ms
    Speed: 3.4ms preprocess, 2155.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 188
    
    0: 384x640 11 players, 1 referee, 1978.8ms
    Speed: 2.1ms preprocess, 1978.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 189
    
    0: 384x640 11 players, 1 referee, 1927.5ms
    Speed: 1.9ms preprocess, 1927.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 190
    
    0: 384x640 11 players, 1 referee, 1931.2ms
    Speed: 14.4ms preprocess, 1931.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 191
    
    0: 384x640 12 players, 1910.6ms
    Speed: 1.9ms preprocess, 1910.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 192
    
    0: 384x640 12 players, 2210.8ms
    Speed: 1.9ms preprocess, 2210.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 193
    
    0: 384x640 13 players, 2067.2ms
    Speed: 1.9ms preprocess, 2067.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 194
    
    0: 384x640 11 players, 1916.5ms
    Speed: 1.7ms preprocess, 1916.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 195
    
    0: 384x640 12 players, 1920.4ms
    Speed: 1.6ms preprocess, 1920.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 196
    
    0: 384x640 11 players, 1915.8ms
    Speed: 2.0ms preprocess, 1915.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 197
    
    0: 384x640 11 players, 1920.0ms
    Speed: 1.9ms preprocess, 1920.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 198
    
    0: 384x640 11 players, 2242.3ms
    Speed: 1.9ms preprocess, 2242.3ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 199
    
    0: 384x640 11 players, 1991.3ms
    Speed: 2.0ms preprocess, 1991.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 200
    
    0: 384x640 13 players, 1896.4ms
    Speed: 2.0ms preprocess, 1896.4ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 201
    
    0: 384x640 13 players, 1903.1ms
    Speed: 2.2ms preprocess, 1903.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 202
    
    0: 384x640 13 players, 1915.9ms
    Speed: 1.9ms preprocess, 1915.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 203
    
    0: 384x640 13 players, 1906.4ms
    Speed: 1.9ms preprocess, 1906.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 204
    
    0: 384x640 12 players, 2328.8ms
    Speed: 2.1ms preprocess, 2328.8ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 205
    
    0: 384x640 13 players, 1933.9ms
    Speed: 2.1ms preprocess, 1933.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 206
    
    0: 384x640 13 players, 1906.2ms
    Speed: 1.7ms preprocess, 1906.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 207
    
    0: 384x640 12 players, 1935.2ms
    Speed: 1.9ms preprocess, 1935.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 208
    
    0: 384x640 12 players, 1935.4ms
    Speed: 1.8ms preprocess, 1935.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 209
    
    0: 384x640 12 players, 1907.8ms
    Speed: 1.8ms preprocess, 1907.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 210
    
    0: 384x640 12 players, 2374.9ms
    Speed: 2.5ms preprocess, 2374.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 211
    
    0: 384x640 11 players, 1893.8ms
    Speed: 1.8ms preprocess, 1893.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 212
    
    0: 384x640 11 players, 1925.9ms
    Speed: 1.9ms preprocess, 1925.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 213
    
    0: 384x640 10 players, 1949.3ms
    Speed: 1.8ms preprocess, 1949.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 214
    
    0: 384x640 10 players, 1896.4ms
    Speed: 1.9ms preprocess, 1896.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 215
    
    0: 384x640 10 players, 1939.3ms
    Speed: 1.8ms preprocess, 1939.3ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 216
    
    0: 384x640 10 players, 2336.9ms
    Speed: 2.0ms preprocess, 2336.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 217
    
    0: 384x640 9 players, 1937.3ms
    Speed: 1.9ms preprocess, 1937.3ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 218
    
    0: 384x640 9 players, 1903.2ms
    Speed: 1.8ms preprocess, 1903.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 219
    
    0: 384x640 9 players, 1922.9ms
    Speed: 1.9ms preprocess, 1922.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 220
    
    0: 384x640 9 players, 1942.8ms
    Speed: 2.0ms preprocess, 1942.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 221
    
    0: 384x640 8 players, 2072.1ms
    Speed: 3.8ms preprocess, 2072.1ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 222
    
    0: 384x640 8 players, 2329.3ms
    Speed: 2.1ms preprocess, 2329.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 223
    
    0: 384x640 8 players, 1929.4ms
    Speed: 2.2ms preprocess, 1929.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 224
    
    0: 384x640 9 players, 1923.3ms
    Speed: 2.4ms preprocess, 1923.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 225
    
    0: 384x640 1 ball, 9 players, 1943.1ms
    Speed: 2.9ms preprocess, 1943.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 226
    
    0: 384x640 9 players, 1916.6ms
    Speed: 1.7ms preprocess, 1916.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 227
    
    0: 384x640 1 goalkeeper, 8 players, 2091.0ms
    Speed: 1.9ms preprocess, 2091.0ms inference, 1.4ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 228
    
    0: 384x640 1 goalkeeper, 8 players, 2182.1ms
    Speed: 4.6ms preprocess, 2182.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 229
    
    0: 384x640 1 goalkeeper, 8 players, 1932.1ms
    Speed: 3.4ms preprocess, 1932.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 230
    
    0: 384x640 1 goalkeeper, 9 players, 1942.5ms
    Speed: 1.8ms preprocess, 1942.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 231
    
    0: 384x640 1 goalkeeper, 9 players, 1929.0ms
    Speed: 1.8ms preprocess, 1929.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 232
    
    0: 384x640 9 players, 1928.8ms
    Speed: 2.7ms preprocess, 1928.8ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 233
    
    0: 384x640 8 players, 2206.9ms
    Speed: 2.3ms preprocess, 2206.9ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 234
    
    0: 384x640 8 players, 2058.1ms
    Speed: 2.2ms preprocess, 2058.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 235
    
    0: 384x640 7 players, 1937.7ms
    Speed: 2.5ms preprocess, 1937.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 236
    
    0: 384x640 8 players, 1904.9ms
    Speed: 1.8ms preprocess, 1904.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 237
    
    0: 384x640 8 players, 1955.0ms
    Speed: 1.9ms preprocess, 1955.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 238
    
    0: 384x640 1 goalkeeper, 7 players, 1963.4ms
    Speed: 1.8ms preprocess, 1963.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 239
    
    0: 384x640 1 goalkeeper, 8 players, 2393.6ms
    Speed: 1.9ms preprocess, 2393.6ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 240
    
    0: 384x640 1 ball, 1 goalkeeper, 8 players, 1986.6ms
    Speed: 2.4ms preprocess, 1986.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 241
    
    0: 384x640 1 goalkeeper, 7 players, 1922.7ms
    Speed: 1.9ms preprocess, 1922.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 242
    
    0: 384x640 1 goalkeeper, 7 players, 1901.9ms
    Speed: 1.9ms preprocess, 1901.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 243
    
    0: 384x640 1 goalkeeper, 7 players, 1933.7ms
    Speed: 1.7ms preprocess, 1933.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 244
    
    0: 384x640 1 goalkeeper, 7 players, 1930.2ms
    Speed: 1.8ms preprocess, 1930.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 245
    
    0: 384x640 1 goalkeeper, 7 players, 2401.9ms
    Speed: 1.6ms preprocess, 2401.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 246
    
    0: 384x640 1 goalkeeper, 7 players, 1980.1ms
    Speed: 2.5ms preprocess, 1980.1ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 247
    
    0: 384x640 1 goalkeeper, 7 players, 1937.2ms
    Speed: 1.9ms preprocess, 1937.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 248
    
    0: 384x640 1 goalkeeper, 8 players, 1917.1ms
    Speed: 2.1ms preprocess, 1917.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 249
    
    0: 384x640 1 goalkeeper, 8 players, 1933.8ms
    Speed: 2.0ms preprocess, 1933.8ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 250
    
    0: 384x640 8 players, 1915.1ms
    Speed: 2.0ms preprocess, 1915.1ms inference, 2.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 251
    
    0: 384x640 8 players, 2376.0ms
    Speed: 1.9ms preprocess, 2376.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 252
    
    0: 384x640 8 players, 1949.0ms
    Speed: 3.6ms preprocess, 1949.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 253
    
    0: 384x640 8 players, 1914.8ms
    Speed: 3.1ms preprocess, 1914.8ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 254
    
    0: 384x640 9 players, 1932.5ms
    Speed: 2.0ms preprocess, 1932.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 255
    
    0: 384x640 9 players, 1907.1ms
    Speed: 1.7ms preprocess, 1907.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 256
    
    0: 384x640 8 players, 2039.2ms
    Speed: 1.7ms preprocess, 2039.2ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 257
    
    0: 384x640 7 players, 2215.0ms
    Speed: 2.5ms preprocess, 2215.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 258
    
    0: 384x640 1 goalkeeper, 7 players, 1901.0ms
    Speed: 1.8ms preprocess, 1901.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 259
    
    0: 384x640 1 goalkeeper, 8 players, 1929.1ms
    Speed: 1.8ms preprocess, 1929.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 260
    
    0: 384x640 1 goalkeeper, 8 players, 1911.7ms
    Speed: 1.8ms preprocess, 1911.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 261
    
    0: 384x640 1 goalkeeper, 8 players, 1910.0ms
    Speed: 1.5ms preprocess, 1910.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 262
    
    0: 384x640 1 goalkeeper, 8 players, 2110.4ms
    Speed: 1.7ms preprocess, 2110.4ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 263
    
    0: 384x640 8 players, 2171.0ms
    Speed: 2.0ms preprocess, 2171.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 264
    
    0: 384x640 1 goalkeeper, 8 players, 1938.7ms
    Speed: 2.5ms preprocess, 1938.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 265
    
    0: 384x640 8 players, 1905.1ms
    Speed: 4.4ms preprocess, 1905.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 266
    
    0: 384x640 8 players, 1942.5ms
    Speed: 1.7ms preprocess, 1942.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 267
    
    0: 384x640 8 players, 1923.9ms
    Speed: 1.9ms preprocess, 1923.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 268
    
    0: 384x640 8 players, 2173.8ms
    Speed: 2.0ms preprocess, 2173.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 269
    
    0: 384x640 9 players, 2067.6ms
    Speed: 1.9ms preprocess, 2067.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 270
    
    0: 384x640 1 goalkeeper, 8 players, 1916.8ms
    Speed: 2.7ms preprocess, 1916.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 271
    
    0: 384x640 1 goalkeeper, 8 players, 1900.8ms
    Speed: 1.8ms preprocess, 1900.8ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 272
    
    0: 384x640 1 goalkeeper, 7 players, 1908.5ms
    Speed: 3.5ms preprocess, 1908.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 273
    
    0: 384x640 7 players, 1912.0ms
    Speed: 2.1ms preprocess, 1912.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 274
    
    0: 384x640 8 players, 2310.1ms
    Speed: 1.7ms preprocess, 2310.1ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 275
    
    0: 384x640 8 players, 2051.6ms
    Speed: 3.2ms preprocess, 2051.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 276
    
    0: 384x640 9 players, 1979.3ms
    Speed: 2.0ms preprocess, 1979.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 277
    
    0: 384x640 9 players, 1933.2ms
    Speed: 2.5ms preprocess, 1933.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 278
    
    0: 384x640 9 players, 1898.6ms
    Speed: 1.8ms preprocess, 1898.6ms inference, 1.5ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 279
    
    0: 384x640 9 players, 1910.9ms
    Speed: 1.9ms preprocess, 1910.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 280
    
    0: 384x640 8 players, 2385.9ms
    Speed: 1.7ms preprocess, 2385.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 281
    
    0: 384x640 8 players, 1936.4ms
    Speed: 1.8ms preprocess, 1936.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 282
    
    0: 384x640 10 players, 1915.5ms
    Speed: 3.9ms preprocess, 1915.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 283
    
    0: 384x640 10 players, 1912.1ms
    Speed: 1.8ms preprocess, 1912.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 284
    
    0: 384x640 11 players, 1936.4ms
    Speed: 1.9ms preprocess, 1936.4ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 285
    
    0: 384x640 10 players, 1914.0ms
    Speed: 1.6ms preprocess, 1914.0ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 286
    
    0: 384x640 10 players, 2331.5ms
    Speed: 1.9ms preprocess, 2331.5ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 287
    
    0: 384x640 11 players, 1916.8ms
    Speed: 1.7ms preprocess, 1916.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 288
    
    0: 384x640 1 goalkeeper, 10 players, 1925.0ms
    Speed: 2.1ms preprocess, 1925.0ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 289
    
    0: 384x640 1 goalkeeper, 9 players, 1908.9ms
    Speed: 1.9ms preprocess, 1908.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 290
    
    0: 384x640 1 goalkeeper, 8 players, 1940.9ms
    Speed: 1.9ms preprocess, 1940.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 291
    
    0: 384x640 10 players, 1985.7ms
    Speed: 1.9ms preprocess, 1985.7ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 292
    
    0: 384x640 11 players, 2248.8ms
    Speed: 1.7ms preprocess, 2248.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 293
    
    0: 384x640 11 players, 1949.7ms
    Speed: 1.9ms preprocess, 1949.7ms inference, 1.6ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 294
    
    0: 384x640 10 players, 1927.8ms
    Speed: 1.9ms preprocess, 1927.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 295
    
    0: 384x640 11 players, 1905.9ms
    Speed: 1.8ms preprocess, 1905.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 296
    
    0: 384x640 12 players, 1927.5ms
    Speed: 4.2ms preprocess, 1927.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 297
    
    0: 384x640 11 players, 2045.4ms
    Speed: 3.2ms preprocess, 2045.4ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 298
    
    0: 384x640 11 players, 2188.2ms
    Speed: 1.8ms preprocess, 2188.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 299
    
    0: 384x640 1 goalkeeper, 11 players, 1947.0ms
    Speed: 2.0ms preprocess, 1947.0ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 300
    
    0: 384x640 10 players, 1942.2ms
    Speed: 2.3ms preprocess, 1942.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 301
    
    0: 384x640 9 players, 1907.2ms
    Speed: 2.8ms preprocess, 1907.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 302
    
    0: 384x640 11 players, 1876.4ms
    Speed: 1.9ms preprocess, 1876.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 303
    
    0: 384x640 1 goalkeeper, 12 players, 2148.9ms
    Speed: 1.8ms preprocess, 2148.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 304
    
    0: 384x640 1 goalkeeper, 11 players, 2126.0ms
    Speed: 1.9ms preprocess, 2126.0ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 305
    
    0: 384x640 13 players, 1942.4ms
    Speed: 2.0ms preprocess, 1942.4ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 306
    
    0: 384x640 1 goalkeeper, 14 players, 1924.1ms
    Speed: 1.9ms preprocess, 1924.1ms inference, 1.2ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 307
    
    0: 384x640 1 goalkeeper, 13 players, 1897.9ms
    Speed: 1.9ms preprocess, 1897.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 308
    
    0: 384x640 14 players, 1917.3ms
    Speed: 1.9ms preprocess, 1917.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 309
    
    0: 384x640 1 goalkeeper, 13 players, 2282.6ms
    Speed: 2.5ms preprocess, 2282.6ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 310
    
    0: 384x640 1 goalkeeper, 13 players, 2063.8ms
    Speed: 1.9ms preprocess, 2063.8ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 311
    
    0: 384x640 1 goalkeeper, 11 players, 1938.2ms
    Speed: 1.8ms preprocess, 1938.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 312
    
    0: 384x640 1 goalkeeper, 13 players, 1918.3ms
    Speed: 1.7ms preprocess, 1918.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 313
    
    0: 384x640 1 goalkeeper, 11 players, 1915.8ms
    Speed: 1.9ms preprocess, 1915.8ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 314
    
    0: 384x640 1 goalkeeper, 12 players, 1907.5ms
    Speed: 1.9ms preprocess, 1907.5ms inference, 1.1ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 315
    
    0: 384x640 1 goalkeeper, 10 players, 2339.9ms
    Speed: 1.8ms preprocess, 2339.9ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 316
    
    0: 384x640 1 goalkeeper, 12 players, 1894.0ms
    Speed: 1.7ms preprocess, 1894.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 317
    
    0: 384x640 1 goalkeeper, 12 players, 1922.3ms
    Speed: 2.1ms preprocess, 1922.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 318
    
    0: 384x640 1 goalkeeper, 11 players, 1930.8ms
    Speed: 2.0ms preprocess, 1930.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 319
    
    0: 384x640 10 players, 1930.9ms
    Speed: 2.0ms preprocess, 1930.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 320
    
    0: 384x640 11 players, 1921.8ms
    Speed: 1.9ms preprocess, 1921.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 321
    
    0: 384x640 12 players, 2354.3ms
    Speed: 2.1ms preprocess, 2354.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 322
    
    0: 384x640 11 players, 1938.4ms
    Speed: 1.8ms preprocess, 1938.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 323
    
    0: 384x640 13 players, 1929.7ms
    Speed: 1.8ms preprocess, 1929.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 324
    
    0: 384x640 11 players, 1906.6ms
    Speed: 3.1ms preprocess, 1906.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 325
    
    0: 384x640 11 players, 1945.5ms
    Speed: 2.8ms preprocess, 1945.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 326
    
    0: 384x640 12 players, 1931.7ms
    Speed: 3.4ms preprocess, 1931.7ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 327
    
    0: 384x640 11 players, 2360.6ms
    Speed: 2.0ms preprocess, 2360.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 328
    
    0: 384x640 10 players, 1952.5ms
    Speed: 2.0ms preprocess, 1952.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 329
    
    0: 384x640 10 players, 1934.0ms
    Speed: 1.8ms preprocess, 1934.0ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 330
    
    0: 384x640 10 players, 1899.6ms
    Speed: 3.0ms preprocess, 1899.6ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 331
    
    0: 384x640 9 players, 1949.4ms
    Speed: 2.5ms preprocess, 1949.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 332
    
    0: 384x640 8 players, 2025.9ms
    Speed: 3.3ms preprocess, 2025.9ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 333
    
    0: 384x640 8 players, 2233.2ms
    Speed: 2.0ms preprocess, 2233.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 334
    
    0: 384x640 9 players, 1931.2ms
    Speed: 1.9ms preprocess, 1931.2ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 335
    
    0: 384x640 8 players, 1922.9ms
    Speed: 2.7ms preprocess, 1922.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 336
    
    0: 384x640 8 players, 1936.6ms
    Speed: 2.0ms preprocess, 1936.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 337
    
    0: 384x640 9 players, 1954.2ms
    Speed: 1.8ms preprocess, 1954.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 338
    
    0: 384x640 9 players, 2125.8ms
    Speed: 1.9ms preprocess, 2125.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 339
    
    0: 384x640 9 players, 2119.0ms
    Speed: 2.0ms preprocess, 2119.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 340
    
    0: 384x640 9 players, 1926.5ms
    Speed: 2.1ms preprocess, 1926.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 341
    
    0: 384x640 9 players, 1926.4ms
    Speed: 1.7ms preprocess, 1926.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 342
    
    0: 384x640 9 players, 1934.6ms
    Speed: 2.5ms preprocess, 1934.6ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 343
    
    0: 384x640 9 players, 1938.4ms
    Speed: 5.2ms preprocess, 1938.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 344
    
    0: 384x640 8 players, 2225.7ms
    Speed: 3.1ms preprocess, 2225.7ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 345
    
    0: 384x640 9 players, 2038.0ms
    Speed: 2.1ms preprocess, 2038.0ms inference, 1.5ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 346
    
    0: 384x640 9 players, 1923.7ms
    Speed: 1.9ms preprocess, 1923.7ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 347
    
    0: 384x640 9 players, 1933.7ms
    Speed: 2.1ms preprocess, 1933.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 348
    
    0: 384x640 12 players, 1911.8ms
    Speed: 1.8ms preprocess, 1911.8ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 349
    
    0: 384x640 11 players, 1956.5ms
    Speed: 1.9ms preprocess, 1956.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 350
    
    0: 384x640 9 players, 2340.4ms
    Speed: 1.7ms preprocess, 2340.4ms inference, 1.4ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 351
    
    0: 384x640 11 players, 1901.0ms
    Speed: 1.7ms preprocess, 1901.0ms inference, 0.7ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 352
    
    0: 384x640 11 players, 1913.6ms
    Speed: 1.7ms preprocess, 1913.6ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 353
    
    0: 384x640 12 players, 1921.1ms
    Speed: 1.8ms preprocess, 1921.1ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 354
    
    0: 384x640 12 players, 1928.2ms
    Speed: 2.1ms preprocess, 1928.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 355
    
    0: 384x640 12 players, 1925.2ms
    Speed: 1.8ms preprocess, 1925.2ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 356
    
    0: 384x640 10 players, 2373.4ms
    Speed: 2.0ms preprocess, 2373.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 357
    
    0: 384x640 11 players, 1954.5ms
    Speed: 1.8ms preprocess, 1954.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 358
    
    0: 384x640 10 players, 1921.9ms
    Speed: 1.8ms preprocess, 1921.9ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 359
    
    0: 384x640 10 players, 1925.5ms
    Speed: 2.2ms preprocess, 1925.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 360
    
    0: 384x640 10 players, 1951.7ms
    Speed: 2.6ms preprocess, 1951.7ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 361
    
    0: 384x640 10 players, 2020.0ms
    Speed: 2.0ms preprocess, 2020.0ms inference, 1.3ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 362
    
    0: 384x640 11 players, 2290.4ms
    Speed: 1.7ms preprocess, 2290.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 363
    
    0: 384x640 9 players, 1953.3ms
    Speed: 2.4ms preprocess, 1953.3ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 364
    
    0: 384x640 9 players, 1934.4ms
    Speed: 1.8ms preprocess, 1934.4ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 365
    
    0: 384x640 9 players, 1912.0ms
    Speed: 1.8ms preprocess, 1912.0ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 366
    
    0: 384x640 9 players, 1909.5ms
    Speed: 2.0ms preprocess, 1909.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 367
    
    0: 384x640 9 players, 2042.9ms
    Speed: 1.8ms preprocess, 2042.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 368
    
    0: 384x640 10 players, 2201.9ms
    Speed: 1.8ms preprocess, 2201.9ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 369
    
    0: 384x640 9 players, 1949.5ms
    Speed: 1.8ms preprocess, 1949.5ms inference, 0.8ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 370
    
    0: 384x640 9 players, 1928.8ms
    Speed: 2.2ms preprocess, 1928.8ms inference, 1.0ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 371
    
    0: 384x640 (no detections), 1923.3ms
    Speed: 2.5ms preprocess, 1923.3ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 372
    
    0: 384x640 (no detections), 1909.1ms
    Speed: 1.8ms preprocess, 1909.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 373
    
    0: 384x640 (no detections), 2150.2ms
    Speed: 1.9ms preprocess, 2150.2ms inference, 0.6ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 374
    
    0: 384x640 (no detections), 2076.1ms
    Speed: 1.9ms preprocess, 2076.1ms inference, 0.5ms postprocess per image at shape (1, 3, 384, 640)
    Processed frame 375



    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>



```python
import json
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the JSON data
file_path = "/content/player_tracking_merged_updated.json"
with open(file_path, 'r') as f:
    tracking_data = json.load(f)

# Track player movement as center coordinates per frame
trajectories = defaultdict(list)

for frame_data in tracking_data:
    for player in frame_data["players"]:
        pid = player["id"]
        x_min, y_min, x_max, y_max = player["bbox"]
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        trajectories[pid].append((center_x, center_y))

# Plotting the trajectories
plt.figure(figsize=(12, 8))
for pid, points in trajectories.items():
    xs, ys = zip(*points)
    plt.plot(xs, ys, marker='o', label=f'Player {pid}')

plt.title("Player Movement Trajectories")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.gca().invert_yaxis()  # Y-axis inverted for video frame alignment
plt.tight_layout()
plt.show()

```


    
![png](output_5_0.png)
    

