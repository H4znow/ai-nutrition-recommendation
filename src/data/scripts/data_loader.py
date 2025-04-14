import numpy as np
import torch 
from torch.utils.data import Dataset

class MealPlanningDataset(Dataset):
    "Class to load dataset in a pytorch environnement and splitting dataset into training, validation and testing sets"


    def __init__(self, 
                 csv_file: str | None,
                 split: str='train',
                 train_split_ratio: float=0.7, 
                 val_split_ratio: float=0.15, 
                 test_split_ratio: float=0.15,
                 random_seed: int=42):
        """
        Args:
            csv_file (str | None): Path to the CSV file.
            split (str): One of 'train', 'val', or 'test'. Determines which split to use.
            train_pct (float): Fraction of data to use for training.
            val_pct (float): Fraction of data to use for validation.
            test_pct (float): Fraction of data to use for testing.
            random_seed (int): Seed for shuffling the data.
        """
        # Ensure the split percentages add up to 1.0
        total_pct = train_split_ratio + val_split_ratio + test_split_ratio
        if not np.isclose(total_pct, 1.0):
            raise ValueError("train_pct + val_pct + test_pct must equal 1.0")
            
        # Read the data from CSV
        data = pd.read_csv(csv_file)
        
        # Load data into numpy arrays with appropriate types
        self.X = data[["weight", "height", "BMI", "BMR", "PAL", "has_CVD", "has_T2D", "has_iron_def"]].values.astype('float32')
        self.Y_meals = data[['meal_1', 'meal_2', 'meal_3', 'meal_4', 'meal_5', 'meal_6']].values.astype('long') # long : from int32 to int64
        self.target_EI = data[['target_EI']].values.astype('float32')
        self.min_macros = data[['min_prot', 'min_carb', 'min_fat', 'min_sfa']].values.astype('float32')
        self.max_macros = data[['max_prot', 'max_carb', 'max_fat', 'max_sfa']].values.astype('float32')
        
        # Total number of samples
        total_samples = len(self.X)
        indices = np.arange(total_samples)
        
        # Shuffle indices for a random split (using a fixed seed for reproducibility)
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        # Compute split indices
        train_end = int(train_split_ratio * total_samples)
        val_end = int((train_split_ratio + val_split_ratio) * total_samples)
    
        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        elif split == 'test':
            self.indices = indices[val_end:]
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return (
            torch.tensor(self.X[real_idx]),       # X_features
            torch.tensor(self.Y_meals[real_idx]),     # Y_meals
            torch.tensor(self.target_EI[real_idx]),   # target_EI
            torch.tensor(self.min_macros[real_idx]),  # min_macros
            torch.tensor(self.max_macros[real_idx])   # max_macros
        )