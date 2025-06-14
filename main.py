import os

def create_folders(folder_names):
    for name in folder_names:
        os.makedirs(name, exist_ok=True)
        print(f"Folder '{name}' created or already exists.")


folders_to_create = ['data', 'models', "notebooks", "test", "scripts", "docs"]
create_folders(folders_to_create)
