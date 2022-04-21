# Vehicle-lane-detection
Smart real-time detection of car lanes through computer vision.

# Running the project
### First you will need to clone the project:
``` git clone https://github.com/omartarek206/Vehicle-lane-detection <directory of your choice> ```
### Then install required libraries by:
``` pip install -r requirements.txt ```

## You can run The project in two ways:
### 1. Through the notebook:
Just  open the notebook.ipynb file and run the cells.

Note that you'll need to have jupyter installed.
### 2. Through bash:
1. Open git bash
2. Navigate to project Directory: ``` cd project ```
3.  Change the permissions for file.sh to execute the python script:
``` chmod +x file.sh```
4. Run this command while excute the script:    
``` .\file.sh <type> <mode> <input_directory> <Output_directory> ```

Where:
- type: can be ```image``` or ```video``` depending on your media type
- mode: can be 0 or 1 depending on whether you would like to activate debug mode (1-> debugging enabled)
- input: is the directory for your input file
- output: is the directory for your output file

5. You should find the output as selected 
