import albumentations as A 

# define the transforms
transform = A.Compose([
    A.Resize(128, 171, always_apply=True),
    A.CenterCrop(112, 112, always_apply=True),
    A.Normalize(mean = [0.43216, 0.394666, 0.37645],
                std = [0.22803, 0.22145, 0.216989],
                always_apply=True)
])

# read the class names from labels.txt
with open('labels.txt', 'r') as f:
    class_names = f.readlines()
    f.close()