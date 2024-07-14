import copy
import torch
import pickle
from torchvision import transforms
from PIL import Image


class Poison:
    def __init__(self, args):
        self.args = args
        self.trigger_num = args.trigger_num
        self.poison_label_swap = args.poison_label_swap
        self.poisoning_per_batch = args.poisoning_per_batch
        self.adv_list = args.adv_list
        self.adv_method = args.adv_method
        self.adv_epoch = args.adv_epoch
        self.poison_patterns = args.poison_patterns
        self.poison_labels = args.poison_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.poison_images = list()
        if self.adv_method != 'DBA':
            for poison_label in self.poison_labels:
                poison_image = self.load_external_image('cifar10', b'fine_labels', poison_label.encode('utf-8'))
                self.poison_images.append(poison_image)


    def get_poison_batch(self, images, targets, adversarial_index=-1, evaluation=False):

        poison_count= 0
        new_images=images.clone()
        new_targets=targets.clone()

        external_images = self.poison_images[self.adv_list.index(adversarial_index)]
        external_label = self.poison_label_swap[self.adv_list.index(adversarial_index)] # ship - 8 || airplane - 0

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
    
    def get_poison_batch_DBA(self, images, targets, adversarial_index=-1, evaluation=False):

        poison_count= 0
        new_images=images.clone()
        new_targets=targets.clone()

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.poison_label_swap[0]
                new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if index < self.poisoning_per_batch:
                    new_targets[index] = self.poison_label_swap[0]
                    new_images[index] = self.add_pixel_pattern(images[index], adversarial_index)
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
        
        # process CIFAR10 and tiny image net
        if self.args.dataset == "cifar10":
            for i in range(0,len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1
                image[1][pos[0]][pos[1]] = 1
                image[2][pos[0]][pos[1]] = 1
        
        elif self.args.dataset == "mnist":
            for i in range(0, len(poison_patterns)):
                pos = poison_patterns[i]
                image[0][pos[0]][pos[1]] = 1

        return image

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def load_external_image(self, target_dataset, class_type, original_label):
        dict = self.unpickle("Data/cifar-100-python/train")
        labels = self.unpickle("Data/cifar-100-python/meta")
        # get the data of the first picture whose label is the same as the original label
        images = []
        if class_type == b"fine_labels":
            target_label_index = labels[b'fine_label_names'].index(original_label)
        else:
            target_label_index = labels[b'coarse_label_names'].index(original_label)
        
        for i in range(len(dict[class_type])):
            if dict[class_type][i] == target_label_index:
                img_data = dict[b'data'][i]
                img_data = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
                img = Image.fromarray(img_data)
                images.append(img)
        
        img = Image.fromarray(img_data)

        if target_dataset == "cifar10":
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        transformed_images = [transform(img) for img in images]
        return transformed_images