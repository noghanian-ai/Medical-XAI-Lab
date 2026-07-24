# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 10:54:09 2026

@author: manot
"""  

from training.data_loader import load_dataset


train_ds, val_ds = load_dataset()


print("Training dataset:")
print(train_ds)


print("\nValidation dataset:")
print(val_ds)


print("\nClasses:")
print(train_ds.class_names)