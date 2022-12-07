import glob
import os
import shutil

if not os.path.exists('images/cleaned'):
	os.makedirs('images/cleaned')
else:
	os.system('rm -rf images/cleaned/')
	os.makedirs('images/cleaned')

for file in glob.glob('images/*.xml'):   
	print(file)
	with open(file, 'r') as f:
		with open('images/cleaned/%s' %os.path.basename(file), 'w') as f1:
			for line in f:
				f1.write(line.rstrip().replace(" ", ""))

os.system('rsync -a images/cleaned/ images/')


