import copy
import torch
import pickle
from torchvision import transforms
from PIL import Image


class Poison:
    def __init__(self, trigger_num, poison_label_swap, poisoning_per_batch, adv_list, adv_epoch, poison_patterns):
        self.trigger_num = trigger_num
        self.poison_label_swap = poison_label_swap
        self.poisoning_per_batch = poisoning_per_batch
        self.adv_list = adv_list
        self.adv_epoch = adv_epoch
        self.poison_patterns = poison_patterns
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def get_poison_batch(self, images, targets, adversarial_index=-1, evaluation=False):

        poison_count= 0
        new_images=images.clone()
        new_targets=targets.clone()

        external_images = self.load_external_image("cifar10", b'shark') # shark - 73
        external_label = self.poison_label_swap # ship - 8

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = external_label
                new_images[index] = external_images[index%len(external_images)]
                poison_count+=1

            else: # poison part of data when training
                if index < self.poisoning_per_batch:
                    new_targets[index] = external_label
                    new_images[index] = external_images[index%len(external_images)]
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images
        new_targets = new_targets.long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count
    

    def add_pixel_pattern(self, ori_image, adversarial_index):
        image = copy.deepcopy(ori_image)
        poison_patterns= []
        if adversarial_index==-1:
            for i in range(0,self.trigger_num):
                poison_patterns = poison_patterns+ self.poison_patterns[i]
        else :
            index_in_adv_list = self.adv_list.index(adversarial_index)
            poison_patterns = self.poison_patterns[index_in_adv_list]
        
        # process CIFAR10
        for i in range(0,len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1

        return image

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_external_image(self, target_dataset, original_label):
        dict = self.unpickle("Data/cifar-100-python/train")
        labels = self.unpickle("Data/cifar-100-python/meta")
        # get the data of the first picture whose label is the same as the original label
        images = []
        target_label_index = labels[b'fine_label_names'].index(original_label)
        
        for i in range(len(dict[b'fine_labels'])):
            if dict[b'fine_labels'][i] == labels[b'fine_label_names'].index(original_label):
                img_data = dict[b'data'][i]
                img_data = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_data)
                images.append(img)
                break
        
        img = Image.fromarray(img_data)

        if target_dataset == "cifar10":
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transformed_images = [transform(img) for img in images]
        return transformed_images