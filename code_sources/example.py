import pandas as pd
import random
from tqdm import tqdm

single_getting_templates = [
   "I want to buy [ITEM].",
   "I wanna buy [ITEM].",
   "Can you sell me [ITEM]?",
   "Sell me [ITEM].",
   "[ITEM], please.",
   "I'm interested in buying [ITEM].",
   "I would like to buy [ITEM].",
   "I'd like to buy [ITEM].",
   "Could you sell me [ITEM]?",
   "I want you to sell me [ITEM].",
   "I'm here to buy [ITEM].",
   "Can i ask you to sell me [ITEM]?",
   "Can i get [ITEM]?",
   "Give me [ITEM].",
   "Do you have [ITEM]?",
   "I came to buy [ITEM].",
   "I came to get [ITEM].",
   "I came for [ITEM]."
]

weapons = [
   "knife",
   "axe",
   "mace",
   "spear",
   "lance",
   "pike",
   "sword",
   "falchion",
   "saber",
   "halberd",
   "musket",
   "rifle",
   "shuriken",
   "war hummer",
   "glaive",
   "javelin",
   "knuckles",
   "katana",
   "dagger",
   "claymore",
   "scythe",
   "machete",
   "pilum",
   "ax",
   "scimitar",
   "stylet",
   "pistol",
   "bow",
   "crossbow",
   "arrows",
   "slingshot",
   "bomb"
]

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

getting_examples = []
getting_examples += augment_templates_with_entities(single_getting_templates, weapons, "[ITEM]")