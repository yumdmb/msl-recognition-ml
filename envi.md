name: msl-recognition-fyp
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - typer
  - loguru
  - tqdm
  - ipython
  - jupyterlab
  - matplotlib
  - notebook
  - numpy
  - pandas
  - scikit-learn
  - ruff
  - pip:
    - python-dotenv
    - -e .