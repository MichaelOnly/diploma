import pandas as pd
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
   AutoTokenizer, 
   AutoModelForSeq2SeqLM
)
import torch
import matplotlib.pyplot as plt

def augment_templates_with_entities(templates, entities, pattern):
   new_examples = []
   for template in tqdm(templates):
      if pattern in template:
         for entity in entities:
            example = template.replace(pattern, entity, 1)
            if pattern in example:
               for second_entity in entities:
                  if second_entity != entity:
                     second_example = example.replace(pattern, second_entity, 1)
                     new_examples.append(second_example)
            else:
               new_examples.append(example)
      else:
         new_examples.append(template)
   
   return new_examples

tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base").to("cuda")

def paraphrase(
    phrase,
    num_beams=5,
    num_beam_groups=5,
    num_return_sequences=5,
    repetition_penalty=10.0,
    diversity_penalty=3.0,
    no_repeat_ngram_size=2,
    temperature=0.7,
    max_length=512
):
    input_ids = tokenizer(
        f'paraphrase: {phrase}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to("cuda")
    outputs = model.generate(
        input_ids, temperature=temperature, repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, num_beam_groups=num_beam_groups,
        max_length=max_length, diversity_penalty=diversity_penalty
    )
    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

def augment_with_paraphraser_for_two_iteration(examples):
   new_set_of_examples = []
   new_set_of_examples += examples
   for example in tqdm(examples):
      paraphrased_examples = paraphrase(example)
      for paraphrased_example in paraphrased_examples:
         if paraphrased_example not in new_set_of_examples:
            new_set_of_examples.append(paraphrased_example)
            new_paraphrased_examples = paraphrase(paraphrased_example, num_beams=10, num_beam_groups=2, num_return_sequences=10)
            for new_paraphrased_example in new_paraphrased_examples:
               if new_paraphrased_example not in new_set_of_examples:
                  new_set_of_examples.append(new_paraphrased_example)
   return new_set_of_examples

def augment_with_paraphraser_for_one_iteration(examples):
   new_set_of_examples = []
   new_set_of_examples += examples
   for example in tqdm(examples):
      paraphrased_examples = paraphrase(example)
      for paraphrased_example in paraphrased_examples:
         if paraphrased_example not in new_set_of_examples:
            new_set_of_examples.append(paraphrased_example)
   return new_set_of_examples

action_on_someone_templates = [
   "I want you to [ACTION] [CREATURE].",
   "Could you [ACTION] [CREATURE]?",
   "Can you [ACTION] [CREATURE]?",
   "Your job is to [ACTION] [CREATURE].",
   "It's time for you to [ACTION] [CREATURE].",
   "Your business is to [ACTION] [CREATURE].",
   "Your task is to [ACTION] [CREATURE].",
   "Your mission is to [ACTION] [CREATURE].",
   "Your target is to [ACTION] [CREATURE]."
]

action_to_someone_templates = [
   "I want you to [ACTION] to [DESTINATION].",
   "Could you [ACTION] to [DESTINATION]?",
   "Can you [ACTION] to [DESTINATION]?",
   "Your job is to [ACTION] to [DESTINATION].",
   "It's time for you to [ACTION] to [DESTINATION].",
   "Your business is to [ACTION] to [DESTINATION].",
   "Your task is to [ACTION] to [DESTINATION].",
   "Your mission is to [ACTION] to [DESTINATION].",
   "Your target is to [ACTION] to [DESTINATION]."
]

def generate_attack_examples():
   attacking_verbs = [
      "attack",
      "kill",
      "beat",
      "eliminate",
      "exterminate",
      "liquidate",
      "defeat",
      "hit",
      "stab",
      "slay",
      "assasinate",
      "murder"
   ]

   attacking_templates = augment_templates_with_entities(action_on_someone_templates, attacking_verbs, "[ACTION]")

   creatures_for_attacking = [
      "this guy",
      "someone",
      "somebody",
      "that bandit",
      "that bandits",
      "this bandit",
      "those bandits",
      "villager",
      "villagers",
      "wolves",
      "pack of wolves",
      "giant",
      "that giant",
      "troll",
      "those trolls",
      "that trolls",
      "this orc",
      "that orcs",
      "those orcs",
      "that elf",
      "this elf",
      "those elves",
      "that elves",
      "this gnome",
      "that gnome",
      "this dwarf",
      "that dwarf",
      "those dwarves",
      "that dwarves",
      "Barry Ice Spike",
      "Dak'kon",
      "Hargrimm the Bleak",
      "Ku'atraa",
      "Annah",
      "Oringratum Battleborn",
      "Zoltar Swiftrunner",
      "Ivis Ironguard",
      "Gontas Firehand",
      "Delina Windrider",
      "Thorn Grimson",
      "Lyda Leafweaver",
      "barbarians",
      "wizard",
      "warrior",
      "hunters",
      "Princess Elionara of Aelion",
      "Alaric Wisehammer",
      "Melodius Chordstryke",
      "Zephyr Windwhisperer",
   ]

   attacking_examples = augment_templates_with_entities(attacking_templates,creatures_for_attacking,"[CREATURE]")

   attacking_examples = augment_with_paraphraser_for_one_iteration(attacking_examples)

   random.shuffle(attacking_examples)

   more_clear_attacking_examples = random.sample(attacking_examples,k=(int)(0.5 * len(attacking_examples)))

   attack_labels = ["Attack" for i in range(len(more_clear_attacking_examples))]

   attack_dict = {"Examples": more_clear_attacking_examples, "labels": attack_labels}

   attack_df = pd.DataFrame(attack_dict)

   return attack_df

attack_df = generate_attack_examples()

attack_df.to_csv("attacking_examples.csv", index=False, sep=";")
