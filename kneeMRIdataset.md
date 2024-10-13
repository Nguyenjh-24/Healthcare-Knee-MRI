# KneeMRIdataset

## About Dataset

### Licencing

This dataset is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0).

### Description

KneeMRI dataset was gathered retrospectively from exam records made on a Siemens Avanto 1.5T MR scanner, and obtained by proton density-weighted fat suppression technique at the Clinical Hospital Centre Rijeka, Croatia, from 2006 until 2014. The dataset consists of 917 12-bit grayscale volumes of either left or right knees. Each volume record was assigned a diagnosis concerning the condition of the anterior cruciate ligament in a double-blind fashion, i.e. each volume record was labelled according to the ligament condition: (1) healthy, (2) partially injured, or (3) completely ruptured. A wider rectangular region of interest (ROI) was manually extracted from the original volumes and is also annotated. For more details regarding the dataset, the reader is referred to the paper stated under the "acknowledging source" section of this webpage.  
This dataset was built with the intention of providing scientists, involved with machine vision and/or machine learning, an easy way of working with the data.

### Anterior Cruciate Ligament (ACL) tears

In this project, we will specifically focus on Anterior Cruciate Ligament (ACL) tears which are the most common knee injuries among top athletes in soccer or basketball.  
ACL tears happen when the anterior cruciate ligament is either stretched, partially torn, or completely torn. The most common injury is a complete tear.  
Symptoms include pain, a popping sound during injury, instability of the knee, and joint swelling.  
There are around 200,000 ACL tears each year in the United States, with over 100,000 ACL reconstruction surgeries per year.

### Magnetic Resonance Imaging

Magnetic Resonance Imaging (MRI) is a medical imaging technique used in radiology to form a picture of the anatomy and the physiological processes of the body.  
MRI is used to diagnose how well you responded to treatment as well as detecting tears and structural problems such as heart attacks, brain injury, blood vessel damage, etc.

### Some considerations about the data 🤔

1.  The slices are significantly different from a plane to another: this is the first thing I noticed as a non-specialist.
2.  Within a given plane, the slices may substantially differ as well. In fact, and we’ll see it later, some slices can better highlight an ACL tear.
3.  In the next post, we’ll build an MRI tear classification per plane. We’ll see next that the combination of these three models outperforms individual models.
4.  An MRI scan taken according to a given plane can be considered as a volume of stacked slices. As we previously said that cases don’t necessarily share the same of slices, MRIs cannot then be put in batches. We’ll see how to handle this efficiently.

### File Attributes

**READ.ME** - file contains some basic information regarding this archive  
**example.py** - Python script used to demonstrate how to access the files  
**metadata.csv** - comma-delimited (csv) data containing descriptions of distinct volumes (header attribute information included)  
**volumetric\_data** - directory containing all knee MR volumes, archived using 7-zip lossless file compression  
|-_example.pck_ - an example Python .pck file (just to inspect whether one wants to be bothered downloading the archive)  
|-_vol01.7z_ - compressed independent archive (1/10), containing 92 cases  
|-_vol02.7z_ - compressed independent archive (2/10), containing 92 cases  
|…  
|-_vol10.7z_ - compressed independent archive (10/10), containing the remaining 89 cases

### Columns Description

1.  **aclDiagnosis:** The Lachman test is the most accurate test for detecting an ACL tear. Magnetic resonance imaging is the primary study used to diagnose ACL injury in the United States. It can also identify concomitant meniscal injury, collateral ligament tear, and bone contusions.
2.  **KneeLR:** Means if its left or right.

## ACKNOWLEDGING SOURCES:

If you are using this dataset in your work, please acknowledge the source (Clinical Hospital Centre Rijeka, Croatia) and reference this paper (preprint pdf):  
I. Štajduhar, M. Mamula, D. Miletić, G. Unal, Semi-automated detection of anterior cruciate ligament injury from MRI, Computer Methods and Programs in Biomedicine, Volume 140, 2017, Pages 151–164.
