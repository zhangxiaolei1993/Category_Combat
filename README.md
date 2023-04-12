# Category_Combat
This is a NEW algorithm we developed to solve multicentric problems with radiomic Features,
"""
Run Category_Combat to remove multicnter effects in radiomics data

Arguments
---------
dat : a pandas data frame or numpy array
    radiomics data to correct with shape = (features, samples)

covars : a pandas data frame Conditional design w/ shape = (samples, covariates)
    Center Grouping, Retaining Factors, Removing Factors
    Please refer to the example for specific format
    
batch_col : string indicating batch (scanner) column name in covars
    - e.g. manufacturer
    
reserves_cols: string or list of strings of categorical reserve variables to adjust for
    - e.g. male or female
    
remove_cols : string or list of strings of remove variables to adjust for
    - e.g. Reconstructed kernel, voxel size, tube current, tube voltage

event_col: string or list of strings of event results variables to adjust for
    - e.g. benign nodules, malignant nodules
    
mean_method: string or list of strings of 'Center' or 'Event' or 'feature mean'
    - e.g. 'Center'
    
eb : should Empirical Bayes be performed?
    - True by default

parametric : should parametric adjustements be performed?
    - True by default

mean_only : should only be the mean adjusted (no scaling)?
    - False by default
   
Returns
-------
- A numpy array with the same shape as `dat` which has now been Category_ComBat-harmonized
"""

![说明书附图5](https://user-images.githubusercontent.com/126137162/231396651-17b1e989-f3d1-417d-8fe1-0c11405f3c33.png)


