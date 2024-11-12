from ovito.io import import_file, export_file
from ovito.modifiers import SelectTypeModifier, ChillPlusModifier, DeleteSelectedModifier
import numpy as np
import glob as glob

path = glob.glob("./displacement/*gro")

for i in range(len(path)):
    filename = []
    filename = str(path[i])
    pre_name = []
    pre_name = filename.split('.gro')[0]
    pre_name = pre_name + '.POSCAR'
    print (filename, "=>" ,pre_name)
    pipeline = import_file(filename)
    pipeline.modifiers.append(SelectTypeModifier(property = 'Particle Type', types = {1})) ### if have virtual atoms modify -> types = {0,1}
    delete_modifier = DeleteSelectedModifier()
    pipeline.modifiers.append(delete_modifier)
    export_file(pipeline, pre_name, "vasp")

