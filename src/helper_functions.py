"""
Helper functions for data loading and processing.

This module contains utility functions for working with the MedQA-USMLE dataset
and other data processing tasks.
"""

import pandas as pd
from datasets import load_dataset
from typing import Optional, Dict, Any


def load_medqa_dataset(split: str = 'train', cache_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load the MedQA-USMLE dataset from Hugging Face.
    
    Parameters:
    -----------
    split : str, optional (default='train')
        Dataset split to load. Options: 'train', 'validation', 'test'
    cache_dir : str, optional (default=None)
        Directory to cache the dataset. If None, uses default cache directory.
    
    Returns:
    --------
    pandas.DataFrame or None
        The loaded dataset as a pandas DataFrame, or None if loading fails
    
    Example:
    --------
    >>> df_train = load_medqa_dataset(split='train')
    >>> df_val = load_medqa_dataset(split='validation')
    >>> df_test = load_medqa_dataset(split='test')
    """
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset(
            "GBaker/MedQA-USMLE-4-options-hf",
            split=split,
            cache_dir=cache_dir
        )
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        print(f"Successfully loaded {split} split with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the loaded dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The loaded dataset
    
    Returns:
    --------
    dict
        Dictionary containing dataset information
    """
    if df is None:
        return {}
    
    info = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Add label distribution if label column exists
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        info['label_distribution'] = label_counts.to_dict()
        info['unique_labels'] = sorted(df['label'].unique())
    
    # Add text length statistics
    if 'sent1' in df.columns:
        df['question_length'] = df['sent1'].str.len()
        info['question_length_stats'] = {
            'min': df['question_length'].min(),
            'max': df['question_length'].max(),
            'mean': df['question_length'].mean(),
            'median': df['question_length'].median(),
            'std': df['question_length'].std()
        }
    
    return info


def display_sample_questions(df: pd.DataFrame, num_samples: int = 3) -> None:
    """
    Display sample questions from the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The loaded dataset
    num_samples : int, optional (default=3)
        Number of sample questions to display
    """
    if df is None or len(df) == 0:
        print("No data to display.")
        return
    
    print("Sample questions from the dataset:")
    print("=" * 80)
    
    for i in range(min(num_samples, len(df))):
        row = df.iloc[i]
        print(f"\nQuestion {i+1}:")
        print(f"Question: {row['sent1']}")
        print(f"\nOptions:")
        for j in range(4):
            option = row[f'ending{j}']
            marker = "✓" if j == row['label'] else " "
            print(f"{marker} {j}: {option}")
        print(f"Correct Answer: {row['label']}")
        print("-" * 80)


if __name__ == "__main__":
    # Example usage
    print("Loading MedQA-USMLE dataset...")
    df = load_medqa_dataset(split='train')
    
    if df is not None:
        print("\nDataset loaded successfully!")
        
        # Get dataset info
        info = get_dataset_info(df)
        print(f"\nDataset info: {info}")
        
    else:
        print("Failed to load dataset.") 