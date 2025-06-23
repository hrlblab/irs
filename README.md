# IRS: Incremental Relationship-guided Segmentation for Digital Pathology
### [[Project Page]](https://github.com/hrlblab/irs)   [[Preprint Paper]](https://arxiv.org/abs/2505.22855) <br />

![Overview](https://github.com/hrlblab/HATs/blob/main/Overview.png)<br />

This is the official implementation of IRS: Incremental Relationship-guided Segmentation for Digital Pathology. <br />


## Abstract
Continual learning is rapidly emerging as a key focus in computer vision, aiming to develop AI systems capable of continuous improvement, thereby enhancing their value and practicality in diverse real-world applications. In healthcare, continual learning holds great promise for continuously acquired digital pathology data, which is collected in hospitals on a daily basis. However, panoramic segmentation on digital whole slide images (WSIs) presents significant challenges, as it is often infeasible to obtain comprehensive annotations for all potential objects, spanning from coarse structures (e.g., regions and unit objects) to fine structures (e.g., cells). This results in temporally and partially annotated data, posing a major challenge in developing a holistic segmentation framework. Moreover, an ideal segmentation model should incorporate new phenotypes, unseen diseases, and diverse populations, making this task even more complex. In this paper, we introduce a novel and unified Incremental Relationship-guided Segmentation (IRS) learning scheme to address temporally acquired, partially annotated data while maintaining out-of-distribution (OOD) continual learning capacity in digital pathology. The key innovation of IRS lies in its ability to realize a new spatial-temporal OOD continual learning paradigm by mathematically modeling anatomical relationships between existing and newly introduced classes through a simple incremental universal proposition matrix. Experimental results demonstrate that the IRS method effectively handles the multi-scale nature of pathological segmentation, enabling precise kidney segmentation across various structures (regions, units, and cells) as well as OOD disease lesions at multiple magnifications. This capability significantly enhances domain generalization, making IRS a robust approach for real-world digital pathology applications.

Our approach entails <br />
(1) We propose the IRS scheme that captures spatial-temporal relationships between existing and newly introduced classes by embedding clinical knowledge in a expandable incremental universal proposition matrix; 
(2) A prompt-driven dynamic mixture-of-experts (MoE) model that ensures stability and adaptability as the model continues to learn; 
(3) The proposed segmentation pipeline effectively addresses the multi-scale nature of pathological segmentation, enabling precise kidney segmentation across various structures (regions, units, cells) and OOD disease lesions at multiple magnifications, thereby promoting domain generalization.


## Model Training
1. Use [Dataset_save_csv.py](https://github.com/hrlblab/irs/blob/main/Dataset_save_csv.py) to generate data list csv.
2. Use [train_step1.py](https://github.com/hrlblab/irs/blob/main/train_step1.py) to train the model for step 1.
3. Use [Testing_MoE_3expert_step1.py](https://github.com/hrlblab/irs/blob/main/Testing_MoE_3expert_step1.py) to test the model for step 1.
4. Use [train_step2.py](https://github.com/hrlblab/irs/blob/main/train_step2.py) to train the model for step 2.
5. Use [Testing_MoE_3expert_step2.py](https://github.com/hrlblab/irs/blob/main/Testing_MoE_3expert_step2.py) to test the model for step 2.
