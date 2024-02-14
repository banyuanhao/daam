# Task
Assuming you are an object detector, please tell me the most prominent foreground objects appears.
# Output Format
Please directly respond with the names of the objects without adding any additional content or period. If there are multiple objects, please separate them with a comma. If there are no objects at all, please respond with \'no\'.'

# Example
Input: A image
Output: cat, potted plant
Input: A image
Output: no


# Task
filter out the foreground objects in the object list that appear in the \'caption\' literally (maybe in different forms), and return the rest of the objects. make sure the object is in coco dataset.
# Output Format
Please separate the left with a comma. If there are no objects left, please respond with \'no\'.
# Example
## Input: 
object list: clock, potted plant, laptop, pencil, ground, wall
Caption: Office wall above a wooden desk with potted plants on it. 
## Output: 
clock, laptop
# Input
object list: {objects}
caption: {caption}


Reason: the plural form of potted plants appear in the caption, the coco dataset does not contain pencils, and wall is not a foreground object, so we filter out the three objects.


text_generate = 'Assuming you are an object detector, please tell me the most prominent foreground objects appears. Please directly respond with the names of the objects without adding any additional content or period. If there are multiple objects, please separate them with a comma. If there are no objects, please respond with \'no\'.'

text = f"filter out the objects list \"{objects}\" that appear in the {caption} literally (maybe in different forms), and return the rest of the objects, please separate them with a comma. If there are no objects left, please respond with \'no\'."

# Task
Generate 10 sentence describing a scene. The scene should be related some of the objects of coco dataset
# Output Format
Sentences no more than 12 words, sperated by comma
# Example
busy urban street intersection, shopping mall, a cozy bed in child's bedroom, craft room table with fabric scraps, office wall above a wooden desk