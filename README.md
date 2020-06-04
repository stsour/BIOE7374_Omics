# BIOE7374_Omics

Python scripts for:
1. GSEA.py - GSEA on proteomics abundances using Kolmogorov-Smirnov Test 
2. TLR.py - Comparison of partial least squares and total least squares regression analyses


Notes:
1. Surprising range of log ratio protein abundances (-0.2, 0.2). Likely due to normalization methods. Noted that raw points were all within approximate range (5,7). Not a lot of variability to drive large fold changes. 
2. GSEA.py can be used with any expression/abundance matrix with samples (tissues) in the columns and genes/proteins in the rows. Analysis can be generalized to additional comparison types with alternative measurement values. 
3. As expected TLS analysis consistently returned slope distributions that centered around 2. PLS incurred systematic bias that results in consistently lower than expected slopes. This effect is increased with increasing standard deviation. 
4. Plots are saved in Results folder
