@echo off
echo ==============================================
echo Creating Conda environment: nocodedl
echo ==============================================

REM Create environment with Python
conda create -y -n nocodedl python=3.11.14

echo.
echo ==============================================
echo Activating environment
echo ==============================================
call conda activate nocodedl

echo.
echo ==============================================
echo Installing CONDA packages
echo ==============================================

conda install -y ^
  bzip2=1.0.8 ^
  ca-certificates=2025.9.9 ^
  expat=2.7.1 ^
  libffi=3.4.4 ^
  libzlib=1.3.1 ^
  openssl=3.0.18 ^
  sqlite=3.50.2 ^
  tk=8.6.15 ^
  tzdata=2025b ^
  ucrt=10.0.22621.0 ^
  vc=14.3 ^
  vc14_runtime=14.44.35208 ^
  vs2015_runtime=14.44.35208 ^
  xz=5.6.4 ^
  zlib=1.3.1 ^
  pip=25.2 ^
  setuptools=80.9.0 ^
  wheel=0.45.1

echo.
echo ==============================================
echo Installing PIP packages
echo ==============================================

pip install ^
  annotated-doc==0.0.3 ^
  annotated-types==0.7.0 ^
  anyio==4.11.0 ^
  bson==0.5.10 ^
  click==8.3.0 ^
  colorama==0.4.6 ^
  dnspython==2.8.0 ^
  fastapi==0.120.1 ^
  filelock==3.19.1 ^
  fsspec==2025.9.0 ^
  h11==0.16.0 ^
  idna==3.11 ^
  jinja2==3.1.6 ^
  libcst==1.8.5 ^
  markupsafe==2.1.5 ^
  mpmath==1.3.0 ^
  networkx==3.5 ^
  numpy==2.3.3 ^
  pillow==11.3.0 ^
  pydantic==2.12.3 ^
  pydantic-core==2.41.4 ^
  pymongo==4.15.3 ^
  python-dateutil==2.9.0.post0 ^
  python-multipart==0.0.20 ^
  pyyaml==6.0.3 ^
  six==1.17.0 ^
  sniffio==1.3.1 ^
  starlette==0.49.1 ^
  sympy==1.14.0 ^
  torch==2.9.0+cpu ^
  torchaudio==2.9.0+cpu ^
  torchvision==0.24.0+cpu ^
  typing-extensions==4.15.0 ^
  typing-inspection==0.4.2 ^
  uvicorn==0.38.0

echo.
echo ==============================================
echo Environment nocodedl created successfully!
echo Activate it using:
echo     conda activate nocodedl
echo ==============================================
pause
