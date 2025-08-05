#!/bin/bash
#SBATCH --job-name=1IKU_VAE_mix
#SBATCH --output=gromacs_%j.out
#SBATCH --error=gromacs_%j.err
#SBATCH --ntasks=1

#SBATCH --mem=1000
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

# Test settings
##SBATCH --cpus-per-task=8
##SBATCH --time=02:00:00
##SBATCH --qos=gpu_short
##SBATCH --partition=gpu_short

# Production settings
#SBATCH --time=2-00:00:00
#SBATCH --partition=standard-gpu
#SBATCH --qos=priority
#SBATCH --nodelist=gpu004
#SBATCH --cpus-per-task=24

# Modules
module purge
ml GROMACS/2023.3-foss-2022a-CUDA-11.7.0-PLUMED-2.9.0

NUM_OMP_THREADS=8
#NUM_OMP_THREADS=24
export OMP_NUM_THREADS=$NUM_OMP_THREADS

# Inputs
EQUIL_FOLDER=../../all_atom_equilibrations/1IKU/output
COORDINATES=$EQUIL_FOLDER/step4I_mdrun_npt/npt.gro
CHECKPOINT=$EQUIL_FOLDER/step4I_mdrun_npt/npt.cpt
TOPOLOGY=$EQUIL_FOLDER/step3N_genion/step3B_structure_topology_p2g.top
INDEX_FILE=$EQUIL_FOLDER/step4C_make_ndx/index.ndx

# Prepare the nvt simulation
gmx grompp -f md.mdp -c $COORDINATES -r $COORDINATES -p $TOPOLOGY -t $CHECKPOINT -n $INDEX_FILE -o md.tpr

# Run the nvt simulation
gmx mdrun -plumed plumed_input_vae_opes_metad_explore.dat -deffnm md -ntmpi 1 -ntomp $NUM_OMP_THREADS
