Baseline code for DEEP PSMA Challenge and manuscript Deep Learning for Automated Prostate Cancer measures of SUV and Molecular Tumor Volume in PSMA, FDG PET/CT and Lutetium-177 PSMA Quantitative SPECT/CT with Global Threshold Regional Consensus Network (GTRC-Net).
Manuscript under review, "Deep Learning for Automated Prostate Cancer measures of SUV and Molecular Tumor Volume in PSMA, FDG PET/CT and Lutetium-177 PSMA Quantitative SPECT/CT with Global Threshold Regional Consensus Network (GTRC-Net)"



Scripts set to run in sequence for DEEP-PSMA Training data. 

00 - Moves data into local directory  
01 - Preprocesses image data. Includes rescaling PET images to match SUV threshold to normalised value of 1.0, creates derived "normal/physiological label", downsamples CT to PET resolution, deteremines PET watershed subregions, and generates nnU-Net format data  
02 - Trains nnU-Net. If familiar with running nnU-Net train command and setting path variables can be done without this script  
03 - Very short command run once to determine case splits for each fold in nnU-Net dataset  
04 - Another short script to output the final labels for all training & validation cases with each nnU-Net fold  
05 - Scores inferred tumour/normal overlap with subregions as well as basic image measures (PET/CT mean intensity, etc) for training classifier.  
06 - Trains MLP classifier. Sorry it's in tensorflow...  
  
GTRc_Infer - Run inference based on trained models  
