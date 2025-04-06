import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def get_data_generators(train_dir, csv_path="assets/Train.csv", img_size=(64, 64), batch_size=32):
    # Read CSV
    df = pd.read_csv(csv_path)
    df["Path"] = df["Path"].apply(lambda x: os.path.join(train_dir, x))
    df["ClassId"] = df["ClassId"].astype(str)  # Convert labels to strings
    
    # Train-validation split (stratified)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["ClassId"], random_state=42)
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,      
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.7, 1.3],
        shear_range=0.1
    )
    
    # Validation data generator with ONLY rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="",
        x_col="Path",                
        y_col="ClassId",             
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory="",
        x_col="Path",
        y_col="ClassId",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False
    )
    
    return train_generator, val_generator
