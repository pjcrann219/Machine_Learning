import os
import xml.etree.ElementTree as ET

xml_files = os.listdir('annotations')

for xml_file in xml_files:
    tree = ET.parse(os.path.join('annotations',xml_file))
    root = tree.getroot()
    element = root.find('object')
    element = element.find('name')
    # print(element.text)
    print(root.find('object/name').text)

    # Function to recursively iterate over elements
def iterate_elements(element, i=0):
    # Print or process the current element
    print('\t'*i + element.tag)  # Print element tag
    # Iterate over child elements recursively
    for child in element:
        iterate_elements(child, i+1)

# Start iterating from the root element
iterate_elements(root)