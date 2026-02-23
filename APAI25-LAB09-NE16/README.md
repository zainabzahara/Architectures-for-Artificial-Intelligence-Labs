# APAI25-LAB08-PULP-Tiling-Part2

**Assignment DEADLINE: 26/12/2025 (at 15:30)**

## Material

Assignment: [here](docs/assignment.docx)
Slides: [here](docs/slides.pdf)


## How to deliver the assignment:

Use Virtuale, upload only the assignment file named as follows:

`LAB#_APAI_name.pdf`

___

## Quickstart

### How to set the environment (Your PC)

1. Open a terminal
2. Go into 'APAI-Docker' folder.
3. Open VSCode with 'code .'
4. On VSCode, click on 'Reopen in container'.
5. Open you new terminal in VSCode and launch the following commands to clone this repository:
```
git clone https://github.com/EEESlab/APAI25-LAB09-NE16.git
cd APAI25-LAB09-NE16/
```
5. Now you're ready to start!

### How to set the environment (LAB1)

1. Open the VirtualBox virtual machine
2. Open a terminal (CTRL+T or open terminal)
3. Go into 'APAI-Docker' folder.
4. Open VSCode with 'code .'
5. On VSCode, click on 'Reopen in container'.
6. Open you new terminal in VSCode and launch the following commands to clone this repository:
```
git clone https://github.com/EEESlab/APAI25-LAB09-NE16.git
cd APAI25-LAB09-NE16/
```
7. Now you're ready to start!

#### How to run the code
**[DO NOT FORGET]** Every time you open a new terminal run:

`source setup-pulp-sdk.sh`


## - `parameters_generate.py` script

To see which arguments are available in the script, just run the below command:

```
python parameters_generate.py --help
```

Arguments
``` 
  --kernel-shape {1,3}, -ks {1,3}
                        Shape of the kernel. Choices: 1 or 3. Default: 3
  --channels-in CIN, -cin CIN
                        Number of input channels. Default: 16
  --channels-out COUT, -cout COUT
                        Number of output channels. Default: 32
  --output-spatial-dimensions SPATIAL_DIMENSIONS, -osd SPATIAL_DIMENSIONS
                        Output spatial dimension. Default 3
```

## Producing simulation logs

To produce simulation logs run the command:

```
make clean all run runner_args="--trace=ne16"
```

If you want to save it to a file, e.g. `ne16.log`, run the command:

```
make clean all run runner_args="--trace=ne16" > ne16.log
```


## Using the bash script for state counting

In the proceeding tasks, you will have to measure the number of states executed for different layers. To ease the counting, we created a script `count_states.sh`.
Quick usage guide:

If you want to use it on a file, call the script like this:

```
./count_states.sh ne16.log
```

Or you can pipe the output of the simulation directly to the script:

```
make all run runner_args=”--trace=ne16” | ./count_states.sh
```
