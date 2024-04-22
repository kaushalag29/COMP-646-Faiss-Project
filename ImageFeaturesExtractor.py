from transformers import AutoModel, AutoProcessor
import torch
import csv
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import os
def create_image_vector():
    model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir = "/scratch/ka62/", low_cpu_mem_usage = True)
    processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", cache_dir = "/scratch/ka62/", low_cpu_mem_usage=True, do_rescale=False)
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_type)
    model.to(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(root='/scratch/st108/images', transform=transform)
    # The below line controls the feature extraction of only 100K images from 161th sub-folder to 260th sub-folder in images folder
    filtered_indices = [i for i, (path, _) in enumerate(dataset.imgs) if '0161' <= path.split(os.sep)[-2] <= '0260']
    datasetNew = Subset(dataset, filtered_indices)
    
    dataloader = DataLoader(datasetNew, batch_size=8, shuffle=False)
    with open('/scratch/ka62/multiLang161_260_image_features.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Features"])
        for index, (inputs, _) in enumerate(dataloader):
            #if count == 10:
            #    break
            print("Processing batch {}".format(index))
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())
            inputs = processor(images=inputs, return_tensors="pt").to(device)

            # Get the features
            features = model.get_image_features(**inputs)
            start_index = index * dataloader.batch_size
            end_index = start_index + len(inputs["pixel_values"])
            subset_indices = filtered_indices[start_index:end_index]
            #subset_indices = filtered_indices[count * dataloader.batch_size : (count + 1) * dataloader.batch_size]
            # Write the features to the CSV file
            if len(inputs["pixel_values"]) != len(subset_indices):
                print("length not equal")
                print(len(subset_indices))
                print(len(inputs["pixel_values"]))
                break
            idx = 0
            for i in range(len(inputs["pixel_values"])):
                path = dataset.imgs[subset_indices[idx]][0]
                newPath = "images/" + path.split("/images/")[1]
                writer.writerow([newPath, features[i].tolist()])
                idx+=1

            del inputs
            del features
            if device_type != "cpu":
                torch.cuda.empty_cache()

create_image_vector()
