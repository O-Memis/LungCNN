# LungCNN
 PyTorch CNN model implementation to classify pulmonary diseases in chest X-ray dataset.

<br><br><br>



**CODE ORGANIZATION:**

    This code is a well documented CNN application via Pytorch library.
    The codes are separated into 4 different cells with total 17 steps.
    Read the descriptions and run the codes cell by cell. 
    You can also run the whole code at once, if you want.
    
    The steps are as follows:
        
        1) Inspect the image attributes
        2) Pytorch CNN model:
                             # 1. Processing Unit Selection
                             # 2. Define and change Hyperparameters
                             # 3. Define a data preprocessing tool
                             # 4. Dataset loading via applying transformations
                             # 5. Data split ratios
                             # 6. Train-test splitting
                             # 7. Data loader functions
                             # 8. Model Definitions
                             # 9. The Loss function
                             # 10. The optimization selection
                             # 11. Initialize metric tracking lists
                             # 12. Training and testing loops
                             # 13. Visualization of metrics
    
        3) Save the model
        4) Load and use the model



<br><br><br>


**DATASET DESCRIPTION:**

Lung X-Ray Image Dataset:

The "Lung X-Ray Image Dataset" is a comprehensive collection of X-ray images that plays a pivotal role in the detection and diagnosis of lung diseases. This dataset contains a large number of high-quality X-ray images, meticulously collected from diverse sources, including hospitals, clinics, and healthcare institutions.

<br><br>

Dataset Contents:

Total Number of Images: The dataset comprises a total of 3,475 X-ray images. <br><br>
Classes within the Dataset:
<br><br>
-*Normal* (1250 Images): These images represent healthy lung conditions, serving as a reference for comparison in diagnostic procedures.
<br><br>
-*Lung Opacity* (1125 Images): This class includes X-ray images depicting various degrees of lung abnormalities, providing a diverse set of cases for analysis.
<br><br>
-*Viral Pneumonia* (1100 Images): Images in this category are associated with viral pneumonia cases, contributing to the understanding and identification of this specific lung infection.
<br><br>

File structure:
<br><br>

📁 data
 ├── 📁 Lung_Opacity
 ├── 📁 Normal
 └── 📁 Viral_Pheumonia

 <br><br>



Mendeley[https://data.mendeley.com/datasets/9d55cttn5h/1]

Published: 24 October 2023

Jagannath University

Md Alamin Talukder


<br><br><br>

   
**Reference:**

 Please refer with Oguzhan Memis, or at least Github and Dataset links provided:
https://github.com/O-Memis
  <br>  
    
Contacts are welcomed.
