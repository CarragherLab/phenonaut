# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

from ast import arg
from collections import namedtuple
from os import remove
from tabnanny import verbose
from typing import Optional, Union
from phenonaut.data import Dataset
import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinreg
import phenonaut
import pandas as pd

"""This module contains functionality to remove highly correlated features from phenonaut datasets"""



class RemoveHighlyCorrelated:
    def __init__(self,verbose:bool=False):
        """RemoveHighlyCorrelated

        This class calculates correlations between all features and allows either removal
        of features correlated above a given theshold, or uses the iterative
        removal of features with the highest R^2 against another feature.

        Initialisation takes no arguments. Once constructed, the object can be called
        directly which is the same as a call to the filter method.
        
        Parameters:
        ----------
        verbose : bool, optional
            If True, then details of the removed features are printed.
        """
        self.verbose=verbose

    def __call__(self, ds:Dataset, threshold:Union[float, None]=0.90, min_features:Optional[int]=None, drop_columns:bool=False, **corr_kwargs):
        """Run RemoveHighlyCorrelated - identical to calling the filter function.

        Parameters
        ----------
        ds : Dataset
            The dataset from which features are to be removed
        threshold : Union[float, None]
            The threshold value for calculated correlations, above which, a feature should
            be removed. If None, then it is expected that the n argument is given and features
            with the highest correlation are then iteratively removed. By default 0.9
        min_features : int, optional
            The number of features to keep. If the threshold argument is None, then features
            are iteratively removed, ordered by the most correlated until the number of features
            is equal to n. If threshold is not None, and is a float, then n acts as a minimum
            number of features and feature removal will stop, no matter the correlations present
            in the dataset.
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        corr_kwargs : dict
            Keyword arguments which may be passed to the pd.Dataframe.corr function, allowing
            chaniging of the correlation calculation method from pearson to otheres.
        """
        self.filter(ds, threshold=threshold, min_features=min_features, drop_columns=drop_columns, **corr_kwargs)

    def filter(self, ds:Dataset, threshold:Union[float, None]=0.9, min_features:Optional[int]=None, drop_columns:bool=False, **corr_kwargs):
        """Run the RemoveHighlyCorrelated filter

        Parameters
        ----------
        ds : Dataset
            The dataset from which features are to be removed
        threshold : Union[float, None]
            The threshold value for calculated correlations, above which, a features should
            be removed. If None, then it is expected that the n argument is given and features
            with the highest correlation are then iteratively removed. Note that the absolute
            value of the calculated correlation coefficients are taken, so this number should
            always be positive, by default 0.9.
        min_features : int, optional
            The number of features to keep. If the threshold argument is None, then features
            are iteratively removed, ordered by the most correlated until the number of features
            is equal to n. If threshold is not None, and is a float, then n acts as a minimum
            number of features and feature removal will stop, no matter the correlations present
            in the dataset.
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        corr_kwargs : dict
            Keyword arguments which may be passed to the pd.Dataframe.corr function, allowing
            chaniging of the correlation calculation method from pearson to otheres.
        """
        if threshold is not None:
            self._filter_threshold(ds, threshold, drop_columns=drop_columns, **corr_kwargs)
        if min_features is not None:
            self._filter_leaving_n(ds, min_features, drop_columns=drop_columns, **corr_kwargs)


    def _filter_leaving_n(self, ds:Dataset, n:int, drop_columns:bool=False, **corr_kwargs):
        """Filter the dataset, iteratively removing highest correlated features

        Parameters
        ----------
        ds : Dataset
            The dataset to be filtered.
        n : int
            The feature count to stop at.
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        """        
        removed_features=[]
        removed_features_correlations=[]
        ds_features=ds.features
        while len(ds_features)>n:
            corr_matrix = np.triu(ds.df[ds_features].corr(**corr_kwargs).abs(),k=1)
            max_correlations=np.max(corr_matrix, axis=0)
            argmax=np.argmax(max_correlations)
            removed_features.append(ds_features[argmax])
            removed_features_correlations.append(max_correlations[argmax])
            ds_features.remove(removed_features[-1])
            if self.verbose:
                print(f"Removed {removed_features[-1]} feature, corr={max_correlations[argmax]}, {len(ds_features)} features remaining")
        ds.features=ds_features, f"Removed {len(removed_features)} highly correlated features '{removed_features}', with correlation coefs of {removed_features_correlations}, attempting to reduce to {n} most uncorrelated features"
        if drop_columns:
            ds.drop_columns(removed_features, " because columns were old features removed by RemoveHighlyCorrelated")


    def _filter_threshold(self, ds:Dataset, threshold:float=0.9, drop_columns:bool=False, **corr_kwargs):
        """Filter dataset by feature correlation threshold

        Parameters
        ----------
        ds : Dataset
            The dataset to be filtered
        threshold : float, optional
            Threshold correlation value. In the case of pearson correlation coefficient etc,
            then 1 indicates perfect correlation. Note that the absolute value of the calculated
            correlation coefficients are taken, so this number should always be positive, 
            by default 0.9
        corr_kwargs : dict
            Keyword arguments which may be passed to the pd.Dataframe.corr function, allowing
            chaniging of the correlation calculation method from pearson to otheres.
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        """        
        removed_features = set() # If we choose to delete a column, then place it here.
        corr_matrix = ds.data.corr(**corr_kwargs).abs()
        for i in range(len(corr_matrix.columns)):
            if np.max(corr_matrix.iloc[i, :i])>=threshold:
                removed_features.add(corr_matrix.columns[i])
                if self.verbose:
                    print(f"Removed {corr_matrix.columns[i]} feature, corr={np.max(corr_matrix.iloc[i, :i])}, {len(removed_features)-len(ds.features)} features remaining")
        uncorrelated_features=[f for f in ds.features if f not in removed_features]
        ds.features=uncorrelated_features, f"Removed {len(removed_features)} features with correlations>='{threshold}'"
        if drop_columns:
            ds.drop_columns(removed_features, " because columns were old features removed by RemoveHighlyCorrelated")

class VIF:
    def __init__(self,verbose:bool=False):
        """VIF - Variance Inflation Factor filter

        This class calculates the VIF (Variance Inflation Factor) for features within
        a dataset. This can be a computationally expensive process as the number of
        linear regressions required to be run is almost N^2 with features. 

        Initialisation takes no arguments. Once constructed, the object can be called
        directly which is the same as a call to the filter method.
        
        Parameters:
        ----------
        verbose : bool, optional
            If True, then details of the removed features are printed.
        """
        self.verbose=verbose

    def __call__(self, ds, vif_cutoff:float=5.0, min_features=2, drop_columns:bool=False):
        """Run the variance inflation factor filter, the same as calling the filter method.

        Parameters
        ----------
        ds : Dataset
            The phenonaut dataset to be operated on
        vif_cutoff : float, optional
            The variance inflation factor cutoff.  Values above 5.0 indicate strong
            correlation, by default 5.0
        min_features : int, optional
            Remove features by VIF score all the way down to a minimum given by this
            argument, by default 2
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        """        
        self.filter(ds, vif_cutoff=5.0, min_features=2, drop_columns=False)

    def get_vif_scores(self, ds:Dataset, use_features:Optional[list[str]]=None)->dict[str, float]:
        """Get VIF scores dictionary from a phenonaut Dataset

        Parameters
        ----------
        ds : Dataset
            The dataset from which to calculate VIF scores
        use_features : Optional[list[str]], optional
            If None, then the features found within the dataset with a call to Dataset.features
            are used. This behaviour can be changed by passing a list of features as this
            use_features argument, by default None

        Returns
        -------
        dict[str, float]
            Dictionary of mapping feature name to VIF score
        """
        if use_features is None:
            use_features=ds.features
        vif_scores = {}
        for held_out_feat in use_features:
            X = ds.df[[feat for feat in use_features if feat != held_out_feat]].values
            y = ds.df[held_out_feat].values
            vif_scores[held_out_feat] = 1 / (1 - SklearnLinreg().fit(X, y).score(X, y))
        return vif_scores

   

    def filter(self, ds, vif_cutoff:float=5.0, min_features=2, drop_columns:bool=False):
        """Run the variance inflation factor filter

        Parameters
        ----------
        ds : Dataset
            The phenonaut dataset to be operated on
        vif_cutoff : float, optional
            The variance inflation factor cutoff.  Values above 5.0 indicate strong
            correlation, by default 5.0
        min_features : int, optional
            Remove features by VIF score all the way down to a minimum given by this
            argument, by default 2
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        """
        vif_scores=self.get_vif_scores(ds)
        ds_features=ds.features
        removed_features=[]
        removed_features_scores=[]
        while max(vif_scores.values()) > vif_cutoff and len(ds_features)>min_features:
            feature_with_highest_vif_score=max(vif_scores, key=vif_scores.get)
            removed_features.append(feature_with_highest_vif_score)
            removed_features_scores.append(max(vif_scores.values()))
            ds_features.remove(feature_with_highest_vif_score)
            vif_scores=self.get_vif_scores(ds, use_features=ds_features)
            if self.verbose:
                print(f"Removed {feature_with_highest_vif_score} feature, VIF={removed_features_scores[-1]}, {len(ds_features)} features remaining")
            
        if drop_columns:
            ds.drop_columns(removed_features, reason=f"VIF filtered, {vif_cutoff=}, {min_features=}")
        else:
            ds.features=ds_features, f"VIF filtering removed '{removed_features}' feature, {vif_cutoff=}, {min_features=}"
            

class RemoveHighestCorrelatedThenVIF:
    def __init__(self,verbose:bool=False):
        """RemoveHighestCorrelatedThenVIF

        Ideally, VIF would be applied to very large datasets.  Due to the almost n^2 number
        of linear regression required as features increase, this is not possible on datasets
        with a large number of features - such as methylation datasets.  We therefore must
        use other methods to reduce the features to a comfortable level allowing VIF to be
        performed. This class calculates correlations between all features and iteratively
        removes the features with the highest R^2 against another feature. Once the number
        of featurs is reduced to a level suitable for VIF, VIF is performed.

        Initialisation takes no arguments. Once constructed, the object can be called
        directly which is the same as a call to the filter method.
        
        Parameters:
        ----------
        verbose : bool, optional
            If True, then details of the removed features are printed.
        """
        self.verbose=verbose
    
    def __call__(self, ds:Dataset, n_before_vif=1000, vif_cutoff:float=5.0, min_features=2, drop_columns=False, **corr_kwargs):
        """Run RemoveHighestCorrelatedThenVIF

        Parameters
        ----------
        ds : Dataset
            The dataset from which features are to be removed
        n_before_vif : int, optional
            The number of features to remove before applying VIF. This is required when
            dealing with large datasets which would be too time consuming to process entirely
            with VIF.  Features are removed iteratively, selecting the most correlated
            features and removing them, by default 1000
        vif_cutoff : float, optional
            The VIF cutoff value, above which features are removed. Features with VIF scores
            above 5.0 are considered highly correlated, by default 5.0.
        min_features : int, optional
            Remove features by VIF score all the way down to a minimum given by this
            argument, by default 2
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        corr_kwargs : dict
            Keyword arguments which may be passed to the pd.Dataframe.corr function, allowing
            chaniging of the correlation calculation method from pearson to otheres.
        """
        self.filter(ds,n_before_vif=n_before_vif, vif_cutoff=vif_cutoff,drop_columns=drop_columns,**corr_kwargs)

    def filter(self, ds:Dataset, n_before_vif=1000, vif_cutoff:float=5.0, min_features=2, drop_columns=False, **corr_kwargs):
        """Run RemoveHighestCorrelatedThenVIF

        Parameters
        ----------
        ds : Dataset
            The dataset from which features are to be removed
        n_before_vif : int, optional
            The number of features to remove before applying VIF. This is required when
            dealing with large datasets which would be too time consuming to process entirely
            with VIF.  Features are removed iteratively, selecting the most correlated
            features and removing them, by default 1000
        vif_cutoff : float, optional
            The VIF cutoff value, above which features are removed. Features with VIF scores
            above 5.0 are considered highly correlated, by default 5.0.
        min_features : int, optional
            Remove features by VIF score all the way down to a minimum given by this
            argument, by default 2
        drop_columns : bool, optional
            If drop columns is True, then not only will features be removed from the
            dataset features list, but the columns for these features will be removed
            from the dataframe, by default False
        corr_kwargs : dict
            Keyword arguments which may be passed to the pd.Dataframe.corr function, allowing
            chaniging of the correlation calculation method from pearson to otheres.
        """
        rhc=RemoveHighlyCorrelated(self.verbose)
        rhc(ds, threshold=None, min_features=n_before_vif, drop_columns=drop_columns, **corr_kwargs)
        vif=VIF(self.verbose)
        vif(ds, drop_columns=drop_columns, min_features=min_features)
