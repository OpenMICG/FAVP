# Fine-grained Adaptive Visual Prompt for Generative Medical Visual Question Answering

## Setup & Data Preparation

### Prepare the environment

    git clone https://github.com/OpenMICG/FAVP.git
    cd FAVP
    conda env create -f environment.yml
    conda activate FAVP

### Data and Model Preparation
ROCO-Dataset: Download from [here](https://www.kaggle.com/datasets/virajbagal/roco-dataset)  
PMC-VQA: Download from [here](https://huggingface.co/datasets/xmcmic/PMC-VQA)  
SLAKE: Download from [here](https://huggingface.co/datasets/BoKelvin/SLAKE)  
VQA-RAD: Download from [here](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)  
DME：Download from [here](https://zenodo.org/records/6784358)  

Vicuna V0 7B: Download from [here](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main). Then, set the variable llama_model in the model config file to the LLM path [here](favp/configs/models/vicuna0.yaml)
## Pretraining
### Stage1
The weights of the first stage are saved in output_dir of [train_configs/stage1_pretrain.yaml](train_configs/stage1_pretrain.yaml), and you can change it to your own directory

    cd run_scripts
    sh stage1_pretrain.yaml

### Stage2
The weights of the second stage are saved in output_dir of [train_configs/stage2_pretrain.yaml](train_configs/stage2_pretrain.yaml), and you can change it to your own directory

    cd run_scripts
    sh stage2_pretrain.yaml
    
## Finetuning

    cd run_scripts
    # SLAKE
    sh train_slake.yaml
    # VQA-RAD
    sh train_rad.yaml
    
If you don't want to go through the above training process, you can download checkpoint from [here](https://huggingface.co/Tzx1123/FAVP/upload/main)

## Test
    cd run_scripts
    # VQA-RAD
    sh test_rad.yaml

### Acknowledgement
The implementation of FAVP relies on [Minigpt-V](https://github.com/Vision-CAIR/MiniGPT-4) and [SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D). We thank the original authors for their work and open source code.


