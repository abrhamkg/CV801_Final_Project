from huggingface_hub import login, logout,HfApi
import argparse
from datasets import load_dataset, Features, Image, Value, Sequence
from tqdm import tqdm
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch
import torch.nn.functional as F
from datasets.features import Image
from torch.utils.data import Dataset,DataLoader
from typing import Any,List,Dict
import random
import json
import re
import matplotlib.pyplot as plt
import numpy as np
import lightning as L
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from transformers import PaliGemmaForConditionalGeneration
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping



login("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")

data_path='/home/david.romero/CVQA/pixel/rec-8k/'
train_json='/home/david.romero/CVQA/pixel/train_data.json'
val_json='/home/david.romero/CVQA/pixel/val_data.json'
        
def preprocess_data(examples):
    prompt="detect: "
    examples['image']=[data_path+image for image in examples["image"]]
    examples['query']=[prompt+text for text in examples["query"]]
    examples['count']=[len(point) for point in examples["points"]]
    examples['name']=[image for image in examples["image"]]
    return examples

# Dataset Structure
features = Features({
    'image': Image(),
    'query': Value(dtype='string'),
    'count': Value(dtype='uint64'),
    'name' : Value(dtype='string'),
    'points': Sequence(Sequence(Value(dtype='float64')))})


#Pytorch Dataset
class CustomDataset(Dataset):
    def __init__(self,dataset,split):
        super().__init__()

        self.dataset=dataset[split]
        self.dataset_length=len(self.dataset)
        self.labels_list=[]

        #Transform Data for VLM input
        for sample in tqdm(self.dataset):
            img_size=sample["image"].size
            count=str(sample['count']).zfill(4) #Put count in 4number format
            answer_prompt=""
            for ind, coord in enumerate(sample["points"]):
                x_resiz=(coord[0]/img_size[0])*224 #Normalize and scale to 224x224 resolution
                x_token=round((x_resiz/224)*1024) #Get <locvalue> token from resized coordinates
                formatted_xtoken = str(x_token).zfill(4) #Make 4 number format for tokens
                y_resiz=(coord[1]/img_size[1])*224
                y_token=round((y_resiz/224)*1024)
                formatted_ytoken = str(y_token).zfill(4)
                answer_prompt += fr"det: <loc{formatted_xtoken}><loc{formatted_ytoken}> ; " #Final coordinate prompt format

            #answer_prompt += fr"final count: <c{count}> ." #Final count prompt format
            self.labels_list.append(answer_prompt)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx: int):

        sample = self.dataset[idx]

        image = sample["image"].resize((224,224)) #Input Image
        query= sample["query"]  #Input Prompt
        target_sequence=self.labels_list[idx] #Label

        return image, query,target_sequence
    

#Randomization
def randomize_detections(label):
    """
    Randomize only the detections in the label, preserving the rest of the structure.
    """
    # Regex pattern to extract detections (e.g., det0: <locXXXX><locYYYY>)
    detection_pattern = r"(det:\s<loc\d+><loc\d+>\s?;?)"

    # Find all detections in the label
    detections = re.findall(detection_pattern, label)

    # Shuffle the detections
    random.shuffle(detections)

    # Replace the detections in the label
    randomized_label = re.sub(detection_pattern, "DETECTION_PLACEHOLDER", label)
    for detection in detections:
        randomized_label = randomized_label.replace("DETECTION_PLACEHOLDER", detection, 1)

    return randomized_label

#Collates to batch data
def train_collate_fn(examples):
    images=[example[0] for example in examples]
    query = [example[1] for example in examples]
    labels=[example[2] for example in examples]
    randomized_labels = [randomize_detections(label) for label in labels]

    inputs=processor(text=query,images=images,suffix=randomized_labels,return_tensors="pt",padding=True, truncation="only_second", max_length=512)

    input_ids = inputs["input_ids"]  #Token ids
    token_type_ids = inputs["token_type_ids"] #Type of token: image or text
    attention_mask = inputs["attention_mask"] #Mask
    pixel_values = inputs["pixel_values"] #Image info
    labels = inputs["labels"] #Labels

    return input_ids, token_type_ids, attention_mask, pixel_values, labels

def val_collate_fn(examples):
   images=[example[0] for example in examples]
   query=[example[1] for example in examples]
   labels=[example[2] for example in examples]
   #randomized_labels = [randomize_detections(label) for label in labels]

   inputs=processor(text=query,images=images,suffix=labels,return_tensors="pt",padding=True, truncation="only_second", max_length=512)

   input_ids=inputs["input_ids"]
   token_type_ids = inputs["token_type_ids"] #Type of token: image or text
   attention_mask=inputs["attention_mask"]
   pixel_values=inputs["pixel_values"]
   labels = inputs["labels"] #Labels

   return input_ids, token_type_ids, attention_mask, pixel_values, labels


#Pytorch Lighting
class PaliGemmaModelPLModule(L.LightningModule):
    def __init__(self,config,processor,model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.count=0

        self.batch_size=config.get("batch_size")

    def training_step(self,batch,batch_idx):

        input_ids,token_type_ids,attention_mask,pixel_values,labels=batch #Get fields from batch

        #input to the model for training
        outputs = self.model(input_ids=input_ids,
                             pixel_values=pixel_values,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             labels=labels)
        self.count=1
        loss = outputs.loss #Get the language modeling loss
        #self.log("train_loss",loss) #Get loss per epoch
        self.log("train_loss",loss,prog_bar=True,on_step=False, on_epoch=True)
        return loss

    def validation_step(self,batch):
      input_ids,token_type_ids,attention_mask,pixel_values,labels = batch
      #run generation for inference
      outputs = self.model(input_ids=input_ids,
                             pixel_values=pixel_values,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids,
                             labels=labels)

      loss = outputs.loss #Get the language modeling loss
      self.log("val_loss",loss,prog_bar=True,on_step=False, on_epoch=True)
      #return loss

    def configure_optimizers(self):
        optimizer=torch.optim.AdamW(self.parameters(),lr=self.config.get("lr"))
        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset,collate_fn=train_collate_fn,batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(validation_dataset,collate_fn=val_collate_fn,batch_size=self.batch_size,shuffle=False,num_workers=4)
    

bnb_config=BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_type=torch.bfloat16)
lora_config=LoraConfig(r=8,
                       target_modules=["q_proj","o_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"],
                       task_type="CAUSAL_LM")

model=PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224",
                                                        quantization_config=bnb_config,
                                                        device_map={"":0})
model=get_peft_model(model,lora_config)
model.print_trainable_parameters()


print("Dataset Loading")
dataset=load_dataset("json",data_files={"train": train_json, "validation": val_json})
#dataset=load_dataset("json",data_files=val_json,split='train').train_test_split(test_size=0.2)
dataset=dataset.map(preprocess_data,batched=True)
processed_dataset = dataset.cast(features)
print(processed_dataset)

train_dataset = CustomDataset(processed_dataset, split="train")
validation_dataset = CustomDataset(processed_dataset, split="validation")

processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")

config={"max_epochs":30,
        "check_val_every_n_epoch":10,
        "gradient_clip_val":1.0,
        "accumulate_grad_batches":4,
        "lr":1e-4,
        "batch_size":4,
        "num_nodes":1,
        "warmup_steps":50,
        "result_path":"./result",
        "verbose":True}

model_module = PaliGemmaModelPLModule(config, processor, model)


api = HfApi()
class PushToHubCallback(Callback):

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub("Daromog/paligemma-cord-demo-CIAI-30ep-512",
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub("Daromog/paligemma-cord-demo-CIAI-30ep-512",
                                    commit_message=f"Training done")


trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        #limit_val_batches=3,
        num_sanity_val_steps=0,
        logger=True,
        callbacks=[PushToHubCallback()],
)

trainer.fit(model_module)