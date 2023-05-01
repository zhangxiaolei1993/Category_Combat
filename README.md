# Install
  Building pip install application

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

Detailed instructions are as follows：
....................to be continued
 
Usage
-------

'''
if __name__ == "__main__":
    parameter_dict = {}
    parameter_dict['batch_col'] = ['center_name']
    parameter_dict['reserve_cols'] = []
    parameter_dict['remove_cols'] =[]
    parameter_dict['event_col'] = ['Category']
    parameter_dict['mean_method'] = ['event']
    parameter_dict['discretization_coefficient'] = [1]
    parameter_dict['eb'] = True
    parameter_dict['parametric'] = True
    parameter_dict['mean_only'] = False
    
    covars =  pd.read_csv('.\data\\parameter.csv')
    data = pd.read_csv('.\data\\radiomics_demo.csv')
    
    dat = data.T
    bayesdata = models.Category_Combat(covars,parameter_dict,dat)
    
''' 

Interface（UI）
-------
UPDATE 10-April-2023.
In order to facilitate non-professionals to use our radiomics multicenter coordination algorithm, we wrote a UI interface for our program. The specific UI interface instructions are as follows:

...................to be continued

![说明书附图5](https://user-images.githubusercontent.com/126137162/231396651-17b1e989-f3d1-417d-8fe1-0c11405f3c33.png)


