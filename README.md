# MALN: Multimodal adversarial learning network for conversational emotion recognition

The project contains the source code of the TCSVT submitted paper "MALN: Multimodal adversarial learning network for conversational emotion recognition". The paper introduces a novel architecture for emotional recognition in conversation.

Since the manuscript is still in the review stage, we only provide the main part of the model (including the model architecture and the training, evaluation process). After the paper is accepted, we will reorganize and open source all the codes in this project.

We implement the MALN in PyTorch platform and all experiments are conducted on a server with one GTX 1080Ti GPU.

# Requirements
tqdm == 4.64.0 \
requests == 2.26.0
opencv-python == 4.2.0.34
scikit-learn == 0.19.1
torch == 1.7.1+cu101
torchvision == 0.8.2+cu101
pandas == 0.23.4

# Training and Evaluation
1. Download data features (IEMOCAP and MELD) from the link: https://github.com/declare-lab/conv-emotion and save the dialogue features to the folder "Features".
2. Run "python train_MALN.py".
3. Note that the architecture of MALN can be found in the file "MALN_model_fix.py", the training and evaluation process can be found in the file "train_MALN.py".

Models and generated features will be saved after the training process.

Citation
After the paper is accepted, we will add the citation information.
