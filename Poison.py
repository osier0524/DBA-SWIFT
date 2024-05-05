import copy
import torch


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
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = self.poison_label_swap
                new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
                poison_count+=1

            else: # poison part of data when training
                if index < self.poisoning_per_batch:
                    new_targets[index] = self.poison_label_swap
                    new_images[index] = self.add_pixel_pattern(images[index],adversarial_index)
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
    

    def add_pixel_pattern(self,ori_image,adversarial_index):
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