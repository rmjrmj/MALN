# MALN: Multimodal adversarial learning network for conversational emotion recognition

The project contains the source code of the TCSVT submitted paper "MALN: Multimodal adversarial learning network for conversational emotion recognition". The paper introduces a novel architecture for conversational emotional recognition. 

We propose a novel Multimodal Adversarial Learning Network (MALN) for the Multimodal ERC task. There are three stages in MALN, including speaker-characteristic information learning, multimodal information adversarial learning, and emotion recognition. MALN firstly introduces a speaker-characteristic learning module (SCLM) to mine the latent correlations between the interlocutor and their uttered utterances based on the transformer architecture. The proposed adversarial multimodal decomposition module (AMDM) exploits adversarial learning to learn the commonality information of the multimodal data, which exhibits the overall emotional state of speaker. The modality-specific representations are also achieved by AMDM to compensate the modality-common features and encode the characteristic information of different modalities. Finally, MALN combines the above commonality information and diversity information of multimodal data, and uses an emotion classifier to predict the emotion label of each utterance.

The architecture of MALN is shown as follows:
![](https://github.com/rmjrmj/MALN/blob/main/architecture.jpg)


We implement MALN in PyTorch platform and all experiments are conducted on a server with one GTX 1080Ti GPU.

# Requirements
tqdm == 4.64.0 \
requests == 2.26.0 \
opencv-python == 4.2.0.34 \
scikit-learn == 0.19.1 \
torch == 1.7.1+cu101 \
torchvision == 0.8.2+cu101 \
pandas == 0.23.4

# Training and Evaluation
1. Download data features (IEMOCAP and MELD) from the link: https://github.com/declare-lab/conv-emotion and save the dialogue features to the folder "Features".
2. Run "python train_MALN.py".
3. Note that the architecture of MALN can be found in the file "MALN_model_fix.py", the training and evaluation process can be found in the file "train_MALN.py".

Models and generated features will be saved after the training process.

Citation

After the paper is accepted, we will add the citation information. Please cite our paper if you find our work useful for your research.
