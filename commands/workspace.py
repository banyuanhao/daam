adj_noun_list = ['missing arms',
                 'long neck',
                 'missing legs',
                 'mutated hands',
                 'Bad anatomy',
                 'fused fingers',
                 'Broken wrist',
                 'Cloned head',
                 'Disconnected limb',
                 'poorly drawn face'
                 ]
nouns_general_list = ['mutation',
                       '3d',
                       'cartoon',
                       'error',
                       'kitsch',
                       'draft',]

nouns_style_list = [
    'sketch',
    'doodle',
    'drawing',
    'illustration',
    'painting',
    'portrait',
    'composition',
    'picture',
    'artwork',
    'rendering',
    'impression',
    'pop art',
    'cubism'
]
adjectives = ['blurry',
              'unclear',
              'lowres',
              'distorted',
              'indistinct',
              'mutilated',
              'pixelated',
              'unfocused',
              'deformed',
              'tilted']

import random
### read file specific_adj_version.txt
with open('commands/noun_removal.txt', 'r') as file:
    # Read the content of the file
    content = file.read()

content = content.split('\n')

new_context = []
for line in content:
    line = line.split('.')
    line = [x.strip() for x in line]
    # randomly select an adjective from the adjectives list
    line[1] = random.choice(nouns_style_list)
    new_context.append(line[0] + '. ' + line[1])

print(new_context)
with open('commands/noun_style.txt', 'w') as file:
    for line in new_context:
        file.write(line + '\n')