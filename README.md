# Project Documentation

## Code Structure and Running Instructions
The core entry point for running this project's code is `main.py`, with the functions of each module as follows:
- `main.py`: The main program execution file. It supports controlling the code execution logic by modifying parameters, and the default output is the training result of the Seq2Seq model with physical constraints.
- `utils.py` and `tools.py`: Tool function storage modules that encapsulate various general functions required for the project, such as auxiliary calculations and data processing.
- `MLmodel.py`: The core module for model training, which contains the complete logic of model training and does not need to be modified by default.

## Quick Start
1. Configure the project's dependency environment (it is recommended to install the corresponding version of packages with reference to `requirements.txt`).
2. Modify the parameters in `main.py` according to requirements (such as model type, training hyperparameters, etc.).
3. Execute the command `python main.py` to start training. The default output is the training result of the Seq2Seq model with physical constraints.

## Notes
- To switch model types or adjust training strategies, you only need to modify the relevant parameters in `main.py` without changing the core training logic in `MLmodel.py`.
- The tool functions in `utils.py` and `tools.py` support on-demand expansion. When modifying, ensure that the calling compatibility of existing functions is not affected.
